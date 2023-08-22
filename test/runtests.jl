using Mars
using Test

@testset "minimal test" begin

    k = 3
    μ = [1.0, -0.5, 0.12] 
    β = Vector{Float64}([-1.5, 0.9, 0.0, 0.6, -1.8, 0.45])
    β_ns = Vector{Float64}([0.3333])
    σ = [0.4,  0.8, 0.2] 
    #P = [0.7 0.2; 0.3 0.8]
    P = [0.8 0.05 0.2; 0.1 0.85 0.05; 0.1 0.1 0.75]
    T = 1000

    y, _, X = generate_mars(μ, β, β_ns, σ, P, T, 0)

    @test X isa Matrix{Float64}
    @test y isa Vector{Float64}
    @test size(X)[1] == T

    model = MSModel(y, k, intercept = "switching", 
                        exog_switching_vars = reshape(X[:,2:3],T,2),
                        exog_vars = reshape(X[:,4],T,1))

    @test model.x isa Matrix{Float64}
    @test model.P isa Matrix{Float64}
    @test model.β isa Vector{Vector{Float64}}
    @test !isnan(model.Likelihood) && (model.Likelihood != Inf)
    @test model.σ isa Vector{Float64}

    @test get_std_errors(model) isa Vector{Float64}
    @test expected_duration(model) isa Vector{Float64}
    @test state_coeftable(model, 1) == nothing
    @test transition_mat(model) == nothing
    @test summary_mars(model) == nothing

end

@testset "tests with stochastic component" begin
    
    k = 3
    μ = [1.0, -0.5, 0.12] 
    β = Vector{Float64}([-1.5, 0.9, 0.0])
    β_ns = Vector{Float64}([])
    σ = [0.4,  0.5, 0.2] 
    P = [0.9 0.05 0.1; 0.05 0.85 0.05; 0.05 0.1 0.85]
    T = 1000

    y, s_t, X = generate_mars(μ, β, β_ns, σ, P, T, 0)


    model = MSModel(y, k, intercept = "switching", 
                            exog_switching_vars = reshape(X[:,2],T,1))

    @test all(sort([model.β[i][1] for i in 1:model.k]) .- sort(μ) .< 0.3)
    @test all(sort([model.β[i][2] for i in 1:model.k]) .- sort(β) .< 0.3)

end

@testset "stochastic component - only non-s exogenous" begin
    k = 3
    μ = [1.0, -0.5, 0.12] 
    β = Vector{Float64}([])
    β_ns = Vector{Float64}([0.633])
    σ = [1.7,  0.8, 0.9] 
    P = [0.9 0.05 0.1; 0.05 0.85 0.05; 0.05 0.1 0.85]
    T = 1000

    y, s_t, X = generate_mars(μ, β, β_ns, σ, P, T, 0)

    model = MSModel(y, k, intercept = "switching", 
                            exog_vars = reshape(X[:,2],T,1))

    @test (model.β[1][2] .- β_ns[1]) < 0.3
end

@testset "3 state model every exogenous vars" begin

    k = 3
    μ = [1.0, -0.5, 0.12] 
    β = Vector{Float64}([-1.5, 0.9, 0.0, 0.6, -1.8, 0.45])
    β_ns = Vector{Float64}([0.3333])
    σ = [0.4,  0.5, 0.2] 
    #P = [0.7 0.2; 0.3 0.8]
    P = [0.9 0.05 0.1; 0.05 0.85 0.05; 0.05 0.1 0.85]
    T = 1000

    θ = [σ; μ; β; β_ns; vec(P[1,:]) .*10]

    y, s_t, X = generate_mars(μ, β, β_ns, σ, P, T, 0)


    model = MSModel(y, k, intercept = "switching", 
                            exog_switching_vars = reshape(X[:,2:3],T,2),
                            exog_vars = reshape(X[:,4],T,1))

    @test all(isreal.(model.P))
    @test all(model.P .>= 0)
    @test all(model.P .<= 1)
    @test isapprox(sum(model.P, dims=1), ones(1,3))
    @test all(expected_duration(model) .> 0)
    @test size(filtered_probs(model)) == (T, k)
    @test size(smoothed_probs(model)) == (T, k)

end

@testset "parameter transformation" begin
    
    k = collect(2:5)
    n_β = collect(0:5)
    n_β_ns = collect(0:5)
    intercept = ["switching", "non-switching"]

    using Distributions


    for k_i in k
        for n_β_i in n_β 
            for n_β_ns_i in n_β_ns
                for int in intercept 
                    
                    n_int = int == "switching" ? k_i : 1
                    θ = [rand(k_i); rand(Uniform(-5, 5), n_int); rand(Uniform(-5, 5), n_β_i*k_i); rand(Uniform(-5, 5), n_β_ns_i); rand(k_i*(k_i-1))] 
                    σ, β, P = trans_θ(θ, k_i, n_β_i, n_β_ns_i, int)
                    println("k: $k_i, n_β: $n_β_i, n_β_ns: $n_β_ns_i, intercept: $int")

                    @test size(σ)[1] == k_i
                    @test size(σ)[1] == k_i
                    @test size(β)[1] == k_i
                    @test size(β[1])[1] == n_β_i+ 1 + n_β_ns_i
                    @test size(P)[1] == k_i

                end
            end
        end
    end
end

@testset "Less crucial functions" begin
    @test add_lags([1.0,2.0,3.0,4.0], 1) == [2.0 1.0; 3.0 2.0; 4.0 3.0]
end

