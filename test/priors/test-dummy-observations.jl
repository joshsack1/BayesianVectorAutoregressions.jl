@testset "dummy_minnesota: shapes and weights" begin
    Y_endog = BayesianVectorAutoregressions.get_endogenous(df_p, end_vec_p)
    n = length(end_vec_p)
    λ1, λ3 = 0.3, 1.2
    Yd, Xd = BayesianVectorAutoregressions.dummy_minnesota(
        Y_endog,
        lags_p,
        true;
        λ1 = λ1,
        λ3 = λ3,
    )
    σ_ar = sqrt.(BayesianVectorAutoregressions.ar_residual_variances(Y_endog, lags_p))
    @test size(Yd) == (n * lags_p + n, n)
    @test size(Xd) == (n * lags_p + n, n * lags_p + 1)
    for ℓ in 1:lags_p, j in 1:n
        row = (ℓ - 1) * n + j
        @test Xd[row, 1 + (ℓ - 1) * n + j] ≈ σ_ar[j] * ℓ^λ3 / λ1
        @test count(!iszero, Xd[row, :]) == 1
        # Only own first lags carry a nonzero y-side (Bańbura et al. 2010).
        if ℓ == 1
            @test Yd[row, j] ≈ σ_ar[j] / λ1
            @test count(!iszero, Yd[row, :]) == 1
        else
            @test all(iszero, Yd[row, :])
        end
    end
    # Σ block: y = diag(σ̂), x = 0.
    Σ_rows = (n * lags_p + 1):(n * lags_p + n)
    @test Yd[Σ_rows, :] ≈ Diagonal(σ_ar)
    @test all(iszero, Xd[Σ_rows, :])
end

"""
    random_walk_coefficients(n, lags, include_constant)

The ``k \\times n`` coefficient matrix of a pure random walk in the
`estimate_var` row layout: zero constant row (if present), identity on the
first-lag block, zeros on every higher lag.
"""
function random_walk_coefficients(n, lags, include_constant)
    offset = include_constant ? 1 : 0
    B_rw = zeros(n * lags + offset, n)
    B_rw[(offset + 1):(offset + n), :] = I(n)
    return B_rw
end

@testset "dummy_minnesota: a random walk fits the coefficient dummies exactly" begin
    Y_endog = BayesianVectorAutoregressions.get_endogenous(df_p, end_vec_p)
    n = length(end_vec_p)
    for include_constant in (true, false), lags in (1, lags_p, 4)
        Yd, Xd = BayesianVectorAutoregressions.dummy_minnesota(
            Y_endog,
            lags,
            include_constant;
            λ1 = 0.2,
            λ3 = 1.0,
        )
        coef_rows = 1:(n * lags)
        B_rw = random_walk_coefficients(n, lags, include_constant)
        @test maximum(abs, Yd[coef_rows, :] - Xd[coef_rows, :] * B_rw) < 1e-12
    end
end

@testset "dummy_minnesota: composed prior mean is a stable random walk" begin
    Y_endog = BayesianVectorAutoregressions.get_endogenous(df_p, end_vec_p)
    n = length(end_vec_p)
    σ_ar = sqrt.(BayesianVectorAutoregressions.ar_residual_variances(Y_endog, lags_p))
    λ = (λ1 = 0.2, λ3 = 1.0, λ4 = 1e5, λ_soc = 1.0, λ_dio = 1.0, λ_lr = 1.0)
    for comps in (
        [:minnesota, :dummy_initial_obs],
        [:minnesota, :sum_of_coefficients, :dummy_initial_obs],
    )
        prior = BayesianVectorAutoregressions.normal_wishart_prior(
            Y_endog,
            lags_p,
            end_vec_p,
            true;
            dummy_components = comps,
            λ = λ,
        )
        # Every block here is fit exactly by a random walk, so that is the prior mean.
        @test prior.β0 ≈ random_walk_coefficients(n, lags_p, true) atol = 1e-8
        Φ0 = [prior.β0[(2 + (ℓ - 1) * n):(1 + ℓ * n), :]' for ℓ in 1:lags_p]
        @test companion_spectral_radius(Φ0) ≈ 1.0 atol = 1e-6
        # The Σ block alone supplies S0.
        @test prior.S0 ≈ Diagonal(σ_ar .^ 2) atol = 1e-8
        @test isposdef(Symmetric(prior.S0))
    end
end

@testset "dummy_sum_of_coefficients: one row per variable, own lags equal" begin
    Y_endog = BayesianVectorAutoregressions.get_endogenous(df_p, end_vec_p)
    n = length(end_vec_p)
    λ_soc = 0.7
    Yd, Xd = BayesianVectorAutoregressions.dummy_sum_of_coefficients(
        Y_endog,
        lags_p,
        true;
        λ_soc = λ_soc,
    )
    ȳ = vec(sum(Y_endog[1:lags_p, :]; dims = 1) ./ lags_p)
    @test size(Yd) == (n, n)
    @test size(Xd) == (n, n * lags_p + 1)
    for i in 1:n
        weight = ȳ[i] / λ_soc
        @test Yd[i, i] ≈ weight
        @test Xd[i, 1] == 0.0 # never touches the constant
        for ℓ in 1:lags_p
            @test Xd[i, 1 + (ℓ - 1) * n + i] ≈ weight
        end
    end
end

@testset "dummy_initial_observation: single row, constant identified" begin
    Y_endog = BayesianVectorAutoregressions.get_endogenous(df_p, end_vec_p)
    n = length(end_vec_p)
    λ_dio = 0.9
    Yd, Xd = BayesianVectorAutoregressions.dummy_initial_observation(
        Y_endog,
        lags_p,
        true;
        λ_dio = λ_dio,
    )
    ȳ = vec(sum(Y_endog[1:lags_p, :]; dims = 1) ./ lags_p)
    @test size(Yd) == (1, n)
    @test size(Xd) == (1, n * lags_p + 1)
    @test vec(Yd) ≈ ȳ ./ λ_dio
    @test Xd[1, 1] ≈ 1 / λ_dio
    for ℓ in 1:lags_p, j in 1:n
        @test Xd[1, 1 + (ℓ - 1) * n + j] ≈ ȳ[j] / λ_dio
    end
end

@testset "dummy_long_run with H = I reproduces dummy_sum_of_coefficients" begin
    Y_endog = BayesianVectorAutoregressions.get_endogenous(df_p, end_vec_p)
    n = length(end_vec_p)
    λ = 0.7
    Yd_soc, Xd_soc = BayesianVectorAutoregressions.dummy_sum_of_coefficients(
        Y_endog,
        lags_p,
        true;
        λ_soc = λ,
    )
    Yd_lr, Xd_lr = BayesianVectorAutoregressions.dummy_long_run(
        Y_endog,
        lags_p,
        true,
        Matrix{Float64}(I, n, n);
        λ_lr = λ,
    )
    @test Yd_lr ≈ Yd_soc
    @test Xd_lr ≈ Xd_soc
end

@testset "Composed NormalWishartPrior via dummy_components agrees with a manual stack" begin
    Y_endog = BayesianVectorAutoregressions.get_endogenous(df_p, end_vec_p)
    λ = (λ1 = 0.2, λ3 = 1.0, λ4 = 1e5, λ_soc = 1.0, λ_dio = 1.0, λ_lr = 1.0)
    prior = BayesianVectorAutoregressions.normal_wishart_prior(
        Y_endog,
        lags_p,
        end_vec_p,
        true;
        dummy_components = [:minnesota, :sum_of_coefficients, :dummy_initial_obs],
        λ = λ,
    )
    Yd1, Xd1 = BayesianVectorAutoregressions.dummy_minnesota(
        Y_endog,
        lags_p,
        true;
        λ1 = λ.λ1,
        λ3 = λ.λ3,
    )
    Yd2, Xd2 = BayesianVectorAutoregressions.dummy_sum_of_coefficients(
        Y_endog,
        lags_p,
        true;
        λ_soc = λ.λ_soc,
    )
    Yd3, Xd3 = BayesianVectorAutoregressions.dummy_initial_observation(
        Y_endog,
        lags_p,
        true;
        λ_dio = λ.λ_dio,
    )
    moments =
        BayesianVectorAutoregressions.dummy_gram(vcat(Yd1, Yd2, Yd3), vcat(Xd1, Xd2, Xd3))
    @test prior.β0 ≈ moments.β0
    @test prior.Ω0 ≈ moments.Ω0
    @test prior.S0 ≈ moments.S0
    @test prior.ν0 == moments.ν0
    @test isposdef(Symmetric(prior.S0))
    @test isposdef(Symmetric(prior.Ω0))
end
