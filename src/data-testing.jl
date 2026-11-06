"""
    get_endogenous(df::DataFrame, end_vec::Vector{Symbol})

This function will ask for the variables in the dataframe that are endogenous.
It is a helper function to make the other tests faster computationally.
The vector `end_vec` should be a vector of the column names of all of the
endogenous variables.
"""
function get_endogenous(df::DataFrame, end_vec::Vector{Symbol})
    return hcat([df[!, end_var] for end_var in end_vec]...)
end

"""
    adf_tests(df::DataFrame, end_vec::Vector{Symbol}

For each endogenous variable the dataframe provided, this function will return
a type 1, type 2, and a type 4 ADF test in a vector.
"""
function adf_tests(df::DataFrame, end_vec::Vector{Symbol})
    end_vars = [df[!, end_var] for end_var in end_vec]
    adf_types = [:none, :constant, :trend]
    return [ADFTest.(Ref(end_var), adf_types, Ref(1)) for end_var in end_vars]
end

# MacKinnon, Haug & Michelis (1999) asymptotic critical values, generated with
# MacKinnon's `johdist` program (as tabulated in LeSage's Econometrics Toolbox
# and statsmodels' `coint_tables`). Row m ↔ n − r = m common stochastic
# trends under the null; columns are the 90%, 95%, and 99% quantiles.
# `_NONE` = no deterministic terms; `_CONST` = unrestricted constant in the
# VECM (a linear trend in the levels), the case `include_constant = true` fits.
const JOHANSEN_TRACE_CV_NONE = [
    2.9762 4.1296 6.9406
    10.4741 12.3212 16.3640
    21.7781 24.2761 29.5147
    37.0339 40.1749 46.5716
    56.2839 60.0627 67.6367
    79.5329 83.9383 92.7136
    106.7351 111.7797 121.7375
    137.9954 143.6691 154.7977
    173.2292 179.5199 191.8122
    212.4721 219.4051 232.8291
    255.6732 263.2603 277.9962
    302.9054 311.1288 326.9716
]
const JOHANSEN_TRACE_CV_CONST = [
    2.7055 3.8415 6.6349
    13.4294 15.4943 19.9349
    27.0669 29.7961 35.4628
    44.4929 47.8545 54.6815
    65.8202 69.8189 77.8202
    91.1090 95.7542 104.9637
    120.3673 125.6185 135.9825
    153.6341 159.5290 171.0905
    190.8714 197.3772 210.0366
    232.1030 239.2468 253.2526
    277.3740 285.1402 300.2821
    326.5354 334.9795 351.2150
]
const JOHANSEN_MAX_EIG_CV_NONE = [
    2.9762 4.1296 6.9406
    9.4748 11.2246 15.0923
    15.7175 17.7961 22.2519
    21.8370 24.1592 29.0609
    27.9160 30.4428 35.7359
    33.9271 36.6301 42.2333
    39.9085 42.7679 48.6606
    45.8930 48.8795 55.0335
    51.8528 54.9629 61.3449
    57.7954 61.0404 67.6415
    63.7248 67.0756 73.8856
    69.6513 73.0946 80.0937
]
const JOHANSEN_MAX_EIG_CV_CONST = [
    2.7055 3.8415 6.6349
    12.2971 14.2639 18.5200
    18.8928 21.1314 25.8650
    25.1236 27.5858 32.7172
    31.2379 33.8777 39.3693
    37.2786 40.0763 45.8662
    43.2947 46.2299 52.3069
    49.2855 52.3622 58.6634
    55.2412 58.4332 64.9960
    61.2041 64.5040 71.2525
    67.1307 70.5392 77.4877
    73.0563 76.5734 83.7105
]

"""
    johansen_critical_values(n::Int, include_constant::Bool)

Looks up MacKinnon, Haug & Michelis (1999) critical values for the Johansen
trace and maximum-eigenvalue statistics of an ``n``-variable system. Row
``r+1`` of each returned ``n \\times 3`` matrix holds the 90%, 95%, and 99%
quantiles for ``H_0: \\text{rank} \\le r``, i.e. for ``n-r`` common stochastic
trends; rows with ``n-r > 12`` (beyond the published table) are `NaN`.
`include_constant` selects the unrestricted-constant case (`true`) or the
no-deterministic-terms case (`false`). Returns `(trace_cv, max_eig_cv)`.
"""
function johansen_critical_values(n::Int, include_constant::Bool)
    trace_table = include_constant ? JOHANSEN_TRACE_CV_CONST : JOHANSEN_TRACE_CV_NONE
    max_table = include_constant ? JOHANSEN_MAX_EIG_CV_CONST : JOHANSEN_MAX_EIG_CV_NONE
    trace_cv = fill(NaN, n, 3)
    max_eig_cv = fill(NaN, n, 3)
    for r in 0:(n - 1)
        m = n - r
        if m <= size(trace_table, 1)
            trace_cv[r + 1, :] = trace_table[m, :]
            max_eig_cv[r + 1, :] = max_table[m, :]
        end
    end
    return trace_cv, max_eig_cv
end

"""
    JohansenResult

Output of `johansen_trace_test`: the sufficient statistics for choosing a
cointegrating rank and the estimated cointegrating vectors. Element ``r+1`` of
each statistic vector (and row ``r+1`` of each critical-value matrix) refers
to the null hypothesis ``H_0: \\text{rank} \\le r``, ``r = 0, \\ldots, n-1``.

- `trace_stats` — ``LR_{trace}(r) = -T_{eff}\\sum_{i=r+1}^{n}\\log(1-\\hat\\lambda_i)``.
- `max_eig_stats` — ``LR_{max}(r) = -T_{eff}\\log(1-\\hat\\lambda_{r+1})``.
- `eigenvals` — ``\\hat\\lambda_1 \\ge \\cdots \\ge \\hat\\lambda_n``, the squared
  canonical correlations between ``\\Delta y_t`` and ``y_{t-1}``.
- `β` — the ``n \\times n`` matrix of eigenvectors; column ``i`` belongs to
  ``\\hat\\lambda_i``, so `β[:, 1:r]` are the estimated cointegrating vectors under
  rank ``r``. Normalized as ``\\hat\\beta' S_{11} \\hat\\beta = I``.
- `trace_cv`, `max_eig_cv` — ``n \\times 3`` MacKinnon, Haug & Michelis (1999)
  critical values at the 90%, 95%, and 99% levels (`NaN` where ``n-r > 12``).
- `T_eff` — the effective sample size ``T-p`` used by every statistic.
- `include_constant` — whether an unrestricted constant was included.

Read it with `@unpack trace_stats, eigenvals, β = result`, or field access.
"""
struct JohansenResult{T<:Real}
    trace_stats::Vector{T}
    max_eig_stats::Vector{T}
    eigenvals::Vector{T}
    β::Matrix{T}
    trace_cv::Matrix{T}
    max_eig_cv::Matrix{T}
    T_eff::Int
    include_constant::Bool
end

function Base.show(io::IO, ::MIME"text/plain", res::JohansenResult)
    n = length(res.eigenvals)
    case = res.include_constant ? "unrestricted constant" : "no deterministic terms"
    println(io, "Johansen cointegration test ($case), T_eff = $(res.T_eff)")
    cells = ["H₀: rank ≤", "trace", "5% cv", "λ-max", "5% cv", "eigenvalue"]
    println(io, join(lpad.(cells, 11)))
    for r in 0:(n - 1)
        row = [
            string(r),
            string(round(res.trace_stats[r + 1]; digits = 2)),
            string(round(res.trace_cv[r + 1, 2]; digits = 2)),
            string(round(res.max_eig_stats[r + 1]; digits = 2)),
            string(round(res.max_eig_cv[r + 1, 2]; digits = 2)),
            string(round(res.eigenvals[r + 1]; digits = 4)),
        ]
        print(io, join(lpad.(row, 11)))
        r < n - 1 && println(io)
    end
end

"""
    johansen_trace_test(df::DataFrame, end_vec::Vector{Symbol}, p::Int; include_constant = true)

Johansen (1988, 1991) reduced-rank test for cointegration among the
endogenous variables, based on the VECM with ``p`` lags in levels,

``\\Delta y_t = \\Pi y_{t-1} + \\sum_{i=1}^{p-1}\\Gamma_i\\,\\Delta y_{t-i} + c + \\varepsilon_t,``

where the constant ``c`` is present (and unrestricted) when
`include_constant = true`. ``\\Delta y_t`` and ``y_{t-1}`` are each
regressed on the short-run terms, giving residuals ``R_{0t}`` and ``R_{1t}``
and moment matrices ``S_{ij} = T_{eff}^{-1}\\sum_t R_{it}R_{jt}'`` with
``T_{eff} = T - p``. The eigenvalues ``\\hat\\lambda_1 \\ge \\cdots \\ge \\hat\\lambda_n``
and eigenvectors ``\\hat\\beta`` solve

``\\left|\\lambda S_{11} - S_{10}S_{00}^{-1}S_{01}\\right| = 0,
\\qquad \\hat\\beta' S_{11}\\hat\\beta = I,``

and the statistics for ``H_0: \\text{rank}(\\Pi) \\le r`` are

``LR_{trace}(r) = -T_{eff}\\sum_{i=r+1}^{n}\\log(1-\\hat\\lambda_i), \\qquad
LR_{max}(r) = -T_{eff}\\log(1-\\hat\\lambda_{r+1}).``

Choose the rank as the first ``r`` whose statistic falls below its critical
value. Critical values are the MacKinnon, Haug & Michelis (1999) asymptotic
quantiles for the matching deterministic case (an unrestricted constant, or
no deterministic terms when `include_constant = false`). Columns of `β` can
be passed as `H` to `dummy_long_run`. Returns a `JohansenResult`.
"""
function johansen_trace_test(
    df::DataFrame,
    end_vec::Vector{Symbol},
    p::Int;
    include_constant = true,
)
    @assert p >= 1 "p is the VAR lag order in levels and must be at least 1"
    Y = float.(get_endogenous(df, end_vec))
    T, n = size(Y)
    T_eff = T - p
    @assert T_eff > n * p + 1 "Too few observations for a Johansen test with $n variables and p = $p"
    # Create lagged differences and levels
    ΔY = diff(Y; dims = 1)
    Y_lag = Y[p:(end - 1), :]
    # Lagged differences for the VAR Part
    ΔY_lags = zeros(eltype(Y), T_eff, n * (p - 1))
    for i in 1:(p - 1)
        ΔY_lags[:, ((i - 1) * n + 1):(i * n)] = ΔY[(p - i):(end - i), :]
    end
    X = include_constant ? hcat(ones(eltype(Y), T_eff), ΔY_lags) : ΔY_lags
    # Residuals from regressing ΔY_t and Y_{t-1} on the short-run terms
    residualize(M) = size(X, 2) > 0 ? M - X * (X \ M) : M
    R0 = residualize(ΔY[p:end, :])
    R1 = residualize(Y_lag)
    # Product moment matrices
    S00 = R0'R0 / T_eff
    S01 = R0'R1 / T_eff
    S11 = R1'R1 / T_eff
    # Generalized symmetric eigenproblem: real eigenvalues, β' S11 β = I
    A = S01' * (S00 \ S01)
    F = eigen(Symmetric((A + A') / 2), Symmetric(S11))
    order = sortperm(F.values; rev = true)
    eigenvals = F.values[order]
    β = F.vectors[:, order]
    # Test statistics for H₀: rank ≤ r, r = 0, …, n − 1
    max_eig_stats = -T_eff .* log1p.(-eigenvals)
    trace_stats = [sum(max_eig_stats[(r + 1):n]) for r in 0:(n - 1)]
    trace_cv, max_eig_cv = johansen_critical_values(n, include_constant)
    return JohansenResult(
        trace_stats,
        max_eig_stats,
        eigenvals,
        β,
        trace_cv,
        max_eig_cv,
        T_eff,
        include_constant,
    )
end
