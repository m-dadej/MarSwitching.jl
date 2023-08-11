using MARS
using Test

@testset "MARS.jl" begin
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
    @test model.Likelihood isa Float64

end

@testset "Less crucial functions" begin
    @test add_lags([1.0,2.0,3.0,4.0], 1) == [2.0 1.0; 3.0 2.0; 4.0 3.0]
end
