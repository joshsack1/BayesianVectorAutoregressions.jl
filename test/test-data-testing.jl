using DataFrames
using LinearAlgebra

"""
    lcg_innovations(T_obs, n; seed = 12345)

Version-stable pseudo-random innovations from the Park–Miller "minimal
standard" linear congruential generator, rescaled from ``U(0,1)`` to mean
zero and unit variance. Used instead of `Random` so that the hard-coded
`urca` reference values below do not depend on Julia's RNG stream, which may
change between Julia versions.
"""
function lcg_innovations(T_obs, n; seed = 12345)
    state = UInt64(seed)
    E = zeros(T_obs, n)
    for i in eachindex(E)
        state = (state * 48271) % 2147483647
        E[i] = (state / 2147483647 - 0.5) * sqrt(12)
    end
    return E
end

"""
    cointegrated_system(T_obs)

Three I(1) series with exactly one cointegrating relation: ``x_1`` and
``x_2`` are independent random walks and
``x_3 = 0.5x_1 + 0.5x_2 + u``, with ``u`` a stationary AR(1)
(coefficient 0.5). The true cointegrating vector is ``(0.5, 0.5, -1)``.
"""
function cointegrated_system(T_obs)
    E = lcg_innovations(T_obs, 3)
    x1 = cumsum(E[:, 1])
    x2 = cumsum(E[:, 2])
    u = zeros(T_obs)
    for t in 2:T_obs
        u[t] = 0.5 * u[t - 1] + E[t, 3]
    end
    return DataFrame(x1 = x1, x2 = x2, x3 = 0.5 .* x1 .+ 0.5 .* x2 .+ u)
end

df_j = cointegrated_system(400)
end_vec_j = [:x1, :x2, :x3]

@testset "johansen_trace_test: picks rank 1 on a cointegrated system" begin
    res = johansen_trace_test(df_j, end_vec_j, 2)
    @test res isa JohansenResult
    @test issorted(res.trace_stats; rev = true)
    # Reject rank ≤ 0, fail to reject rank ≤ 1, at 5%, for both statistics.
    @test res.trace_stats[1] > res.trace_cv[1, 2]
    @test res.trace_stats[2] < res.trace_cv[2, 2]
    @test res.max_eig_stats[1] > res.max_eig_cv[1, 2]
    @test res.max_eig_stats[2] < res.max_eig_cv[2, 2]
    # The leading eigenvector recovers the true relation up to scale.
    b = res.β[:, 1] ./ -res.β[3, 1]
    @test b ≈ [0.5, 0.5, -1.0] atol = 0.05
end

@testset "johansen_trace_test: statistic identities and normalization" begin
    n = length(end_vec_j)
    for (p, include_constant) in ((1, true), (2, true), (3, false), (1, false))
        res = johansen_trace_test(df_j, end_vec_j, p; include_constant = include_constant)
        @test res.T_eff == nrow(df_j) - p
        @test res.include_constant == include_constant
        @test length(res.trace_stats) == length(res.max_eig_stats) == n
        @test size(res.β) == (n, n)
        @test issorted(res.eigenvals; rev = true)
        @test all(-1e-12 .<= res.eigenvals .< 1)
        for r in 0:(n - 1)
            @test res.trace_stats[r + 1] ≈ sum(res.max_eig_stats[(r + 1):n])
        end
        @test res.trace_stats[n] ≈ res.max_eig_stats[n]
    end
    # With p = 1 and a constant, the short-run regression is just demeaning,
    # so S11 can be rebuilt directly to check β' S11 β = I.
    res = johansen_trace_test(df_j, end_vec_j, 1)
    Y = Matrix(df_j)
    Y_lag = Y[1:(end - 1), :]
    R1 = Y_lag .- sum(Y_lag; dims = 1) ./ size(Y_lag, 1)
    S11 = R1'R1 / res.T_eff
    @test res.β' * S11 * res.β ≈ I(n) atol = 1e-8
end

@testset "johansen_trace_test: matches urca::ca.jo" begin
    # Reference values from R 4 / urca 1.3.4 on the same data, written to CSV
    # with `repr` at full precision:
    #   ca.jo(x, type = "trace", ecdet = "none", K = K, spec = "transitory")
    #   ca.jo(x, type = "eigen", ecdet = "none", K = K, spec = "transitory")
    # `ecdet = "none"` is urca's unrestricted-constant case — the one
    # `include_constant = true` fits — and urca's statistics, like ours, are
    # scaled by T - K. urca reports r = n-1, …, 0, so they are reversed here.
    # (urca's own critical-value tables are not used: they are not the
    # unrestricted-constant quantiles.)
    reference = Dict(
        2 => (
            λ = [0.181386599289, 0.0138972964918, 0.000740429962639],
            trace = [85.5217691762, 5.86471795441, 0.294800278083],
            max = [79.6570512218, 5.56991767633, 0.294800278083],
        ),
        3 => (
            λ = [0.170089160761, 0.012637473445, 0.00098373126792],
            trace = [79.4552733287, 5.43978172203, 0.390733533287],
            max = [74.0154916067, 5.04904818874, 0.390733533287],
        ),
    )
    for (K, ref) in reference
        res = johansen_trace_test(df_j, end_vec_j, K)
        @test res.eigenvals ≈ ref.λ rtol = 1e-9
        @test res.trace_stats ≈ ref.trace rtol = 1e-9
        @test res.max_eig_stats ≈ ref.max rtol = 1e-9
    end
end

@testset "johansen_trace_test: critical values" begin
    res = johansen_trace_test(df_j, end_vec_j, 2)
    # MacKinnon, Haug & Michelis (1999), unrestricted constant, 5%; row r+1 ↔ n - r.
    @test res.trace_cv[:, 2] == [29.7961, 15.4943, 3.8415]
    @test res.max_eig_cv[:, 2] == [21.1314, 14.2639, 3.8415]
    res0 = johansen_trace_test(df_j, end_vec_j, 2; include_constant = false)
    @test res0.trace_cv[:, 2] == [24.2761, 12.3212, 4.1296]
    @test res0.max_eig_cv[:, 2] == [17.7961, 11.2246, 4.1296]
    # Close to the Osterwald-Lenum (1992) Table 1 5% values for n - r = 1…6
    # (finite-sample simulations, so they differ slightly from MHM).
    trace_cv6, _ = BayesianVectorAutoregressions.johansen_critical_values(6, true)
    osterwald_lenum = [3.76, 15.41, 29.68, 47.21, 68.52, 94.15]
    @test all(isapprox.(reverse(trace_cv6[:, 2]), osterwald_lenum; rtol = 0.03))
    # Beyond the 12-trend table the critical value is NaN.
    trace_cv13, max_cv13 = BayesianVectorAutoregressions.johansen_critical_values(13, true)
    @test all(isnan, trace_cv13[1, :])
    @test all(isnan, max_cv13[1, :])
    @test !any(isnan, trace_cv13[2:end, :])
end

@testset "johansen_trace_test: input checks and display" begin
    @test_throws AssertionError johansen_trace_test(df_j, end_vec_j, 0)
    @test_throws AssertionError johansen_trace_test(df_j[1:5, :], end_vec_j, 2)
    res = johansen_trace_test(df_j, end_vec_j, 2)
    shown = sprint(show, MIME"text/plain"(), res)
    @test occursin("unrestricted constant", shown)
    @test occursin("T_eff = $(res.T_eff)", shown)
end
