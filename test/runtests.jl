using BayesianVectorAutoregressions
using Test

@testset "BayesianVectorAutoregressions.jl" begin
    include("test-data-testing.jl")
    include("test-var-estimation.jl")
    include("test-priors.jl")
    include("test-bayesian-estimation.jl")
    include("test-structural-identification.jl")
end
