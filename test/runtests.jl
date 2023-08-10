using MARS
using Test

@testset "MARS.jl" begin
    # Write your tests here.
    @test generate_mars([1.0,2.0], [0.1, 0.5], [0.8 0.1; 0.2 0.9], 20, 0)[1] isa Vector{Float64}
    @test size(generate_mars([1.0,2.0], [0.1, 0.5], [0.8 0.1; 0.2 0.9], 20, 0)[1])[1] == 20
end
