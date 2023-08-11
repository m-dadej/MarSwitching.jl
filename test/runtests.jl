using MARS
using Test

@testset "minimal test" begin
    # Write your tests here.
    @test generate_mars([1.0,2.0], [0.1, 0.5], [0.8 0.1; 0.2 0.9], 20, 0)[1] isa Vector{Float64}
    @test size(generate_mars([1.0,2.0], [0.1, 0.5], [0.8 0.1; 0.2 0.9], 20, 0)[1])[1] == 20

    k = 3
    p = 0
    μ = [1.0, -0.5, 2.0] 
    σ = [0.8,  1.5, 0.5] 
    P = [0.7 0.2; 0.3 0.8]
    P = [0.7 0.15 0.2; 0.2 0.75 0.15; 0.1 0.1 0.65] #[0.8 0.1; 0.2 0.9]  #
    T = 100
    
    X, s_t = generate_mars(μ, σ, P, T+p, 10)

    model = MSModel(X,k, p) 

    @test model.x isa Matrix{Float64}
    @test model.P isa Matrix{Float64}
    @test model.β isa Vector{Vector{Float64}}
    @test !isnan(model.Likelihood) && (model.Likelihood != Inf)
    @test model.σ isa Vector{Float64}
    @test model.rawP isa Vector{Float64}

    @test get_std_errors(model) isa Vector{Float64}
    @test expected_duration(model) isa Vector{Float64}
    @test state_coeftable(model, 1) == nothing
    @test transition_mat(model) == nothing
    @test summary_mars(model) == nothing

end

@testset begin
    k = 3
    p = 0
    μ = [1.0, -0.5, 2.0] 
    σ = [0.2,  0.4, 0.2] 
    P = [0.7 0.2; 0.3 0.8]
    P = [0.8 0.05 0.2; 0.1 0.85 0.05; 0.1 0.1 0.75]
    T = 1000
    
    X, s_t = generate_mars(μ, σ, P, T+p, 10)

    model = MSModel(X,k, p) 

    for i in 1:k 
        @test (sort(vcat(model.β...)) .- sort(μ))[1] < 0.4
    end

    @test all(isreal.(model.P))
    @test all(model.P .>= 0)
    @test all(model.P .<= 1)
    @test all(sum(model.P, dims=1) .== 1)
    @test all(expected_duration(model) .> 0)

end

@testset "Less crucial functions" begin
    @test add_lags([1.0,2.0,3.0,4.0], 1) == [2.0 1.0; 3.0 2.0; 4.0 3.0]
end

