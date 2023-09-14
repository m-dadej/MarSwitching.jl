using MarSwitching
using Test
using StatsBase
using LinearAlgebra

n_rnd_search = 5

@testset "minimal test" begin

    k = 3
    μ = [1.0, -0.5, 0.12] 
    β = Vector{Float64}([-1.5, 0.9, 0.0, 0.6, -1.8, 0.45])
    β_ns = Vector{Float64}([0.3333])
    σ = [0.4,  0.8, 0.2] 
    #P = [0.7 0.2; 0.3 0.8]
    P = [0.8 0.05 0.2; 0.1 0.85 0.05; 0.1 0.1 0.75]
    T = 1000

    y, _, X = generate_msm(μ, σ, P, T, β = β, β_ns = β_ns)

    @test X isa Matrix{Float64}
    @test y isa Vector{Float64}
    @test size(X)[1] == T

    @test MSModel(y,k) isa MSM   

    model = MSModel(y, k, intercept = "switching", 
                        exog_switching_vars = reshape(X[:,2:3],T,2),
                        exog_vars = reshape(X[:,4],T,1))

    @test MarSwitching.loglik(model.raw_params, model.x, k,  model.n_β, model.n_β_ns, model.intercept, model.switching_var)[1] == model.Likelihood                        

    @test model.nlopt_msg == :XTOL_REACHED
    @test isnothing(display(model))
    @test model.x isa Matrix{Float64}
    @test model.P isa Matrix{Float64}
    @test model.β isa Vector{Vector{Float64}}
    @test !isnan(model.Likelihood) && (model.Likelihood != Inf)
    @test model.σ isa Vector{Float64}

    @test size(get_std_errors(model))[1] == size(model.raw_params)[1]
    @test expected_duration(model) isa Vector{Float64}
    @test isnothing(state_coeftable(model, 1))
    @test isnothing(transition_mat(model))
    @test isnothing(summary_msm(model))
    @test isnothing(MarSwitching.check_args(model))

    @test MarSwitching.convert_arg(:exog_vars, exog_vars = rand(100)) isa Matrix{Float64}

end

@testset "stochastic component μ, β + generate_msm(model)" begin
    
    k = 3
    μ = [1.0, -0.5, 0.12] 
    β = Vector{Float64}([-1.5, 0.9, 0.0])
    σ = [0.4,  0.5, 0.2] 
    P = [0.9 0.05 0.1; 0.05 0.85 0.05; 0.05 0.1 0.85]
    T = 1000

    y, s_t, X = generate_msm(μ, σ, P, T, β = β)


    model = MSModel(y, k, intercept = "switching", 
                            exog_switching_vars = reshape(X[:,2],T,1),
                            random_search = n_rnd_search)

    @test model.nlopt_msg == :XTOL_REACHED                    
    @test all(abs.(sort([model.β[i][1] for i in 1:model.k]) .- sort(μ)) .< 0.3)
    @test all(abs.(sort([model.β[i][2] for i in 1:model.k]) .- sort(β)) .< 0.3)
    @test maximum(cor([filtered_probs(model) (s_t .== 3)])[1:3,end]) > 0.6 
    @test maximum(cor([smoothed_probs(model) (s_t .== 3)])[1:3,end]) > 0.7 

    y_, s_t_, X_ = generate_msm(model, 1000)

    model_ = MSModel(y_, k, intercept = "switching", 
                            exog_switching_vars = reshape(X_[:,2],T,1))

    @test all(abs.(sort([model_.β[i][1] for i in 1:model.k]) .- sort(μ)) .< 0.3)
    @test all(abs.(sort([model_.β[i][2] for i in 1:model.k]) .- sort(β)) .< 0.3)
                        
    @test sort(unique(generate_msm(model, 1000)[2])) == collect(1:k)
end

@testset begin "stochastic component - non-switching intercept"
    k = 3
    μ = [1.5, 1.5, 1.5] 
    β_ns = Vector{Float64}([0.7])
    σ = [0.3,  0.6, 0.8] 
    P = [0.9 0.05 0.1; 0.05 0.85 0.05; 0.05 0.1 0.85]
    T = 1000

    y, s_t, X = generate_msm(μ, σ, P, T, β_ns = β_ns)

    model = MSModel(y, k, intercept = "non-switching", 
                            exog_vars = reshape(X[:,2],T,1),
                            random_search = n_rnd_search) 

    @test all([model.β[s][1] for s in 1:model.k] .== model.β[1][1])   
    @test abs(model.β[1][1] - μ[1]) < 0.1                         
end

@testset "stochastic component - only non-s exogenous" begin
    k = 3
    μ = [1.0, -0.5, 0.12] 
    β_ns = Vector{Float64}([0.633])
    σ = [1.7,  0.8, 0.9] 
    P = [0.9 0.05 0.1; 0.05 0.85 0.05; 0.05 0.1 0.85]
    T = 1000

    y, s_t, X = generate_msm(μ, σ, P, T, β_ns = β_ns)

    model = MSModel(y, k, intercept = "switching", 
                            exog_vars = reshape(X[:,2],T,1),
                            random_search = n_rnd_search)

                          
    @test abs.(model.β[1][2] .- β_ns[1]) < 0.3
    @test model.nlopt_msg == :XTOL_REACHED
end

@testset "stochastic component - no intercept model" begin
    k = 3
    μ = [0.0, 0.0, 0.0] 
    β = Vector{Float64}([-1.5, 0.9, 0.0])
    β_ns = Vector{Float64}([-0.33])
    σ = [0.4,  0.3, 0.1] 
    P = [0.9 0.05 0.1; 0.05 0.85 0.05; 0.05 0.1 0.85]
    T = 2000

    y, s_t, X = generate_msm(μ, σ, P, T, β = β, β_ns = β_ns)

    model = MSModel(y, k, intercept = "no", exog_switching_vars = reshape(X[:,2], T, 1),
                            exog_vars = reshape(X[:,3], T, 1),
                            random_search = n_rnd_search)

    @test maximum(cor([filtered_probs(model) (s_t .== 3)])[1:3,end]) > 0.5 
    @test maximum(cor([smoothed_probs(model) (s_t .== 3)])[1:3,end]) > 0.5 

    @test all([model.β[i][1] == 0 for i in 1:model.k])
    @test model.nlopt_msg == :XTOL_REACHED
end

@testset "stochastic component - 3 state model every exogenous vars" begin

    k = 3
    μ = [1.0, -0.5, 0.12] 
    β = Vector{Float64}([-1.5, 0.9, 0.0, 0.6, -1.8, 0.45])
    β_ns = Vector{Float64}([0.3333])
    σ = [0.4,  0.5, 0.2] 
    #P = [0.7 0.2; 0.3 0.8]
    P = [0.9 0.05 0.1; 0.05 0.85 0.05; 0.05 0.1 0.85]
    T = 1000

    y, s_t, X = generate_msm(μ, σ, P, T, β = β, β_ns = β_ns)

    model = MSModel(y, k, intercept = "switching", 
                            exog_switching_vars = reshape(X[:,2:3],T,2),
                            exog_vars = reshape(X[:,4],T,1),
                            random_search = n_rnd_search)

    # to add tests below we need better x0 for P or random search 
    # because around eery 2 estimations the P is very biased                            
    # @test maximum(cor([filtered_probs(model) (s_t .== 3)])[1:3,end]) > 0.6 
    # @test maximum(cor([smoothed_probs(model) (s_t .== 3)])[1:3,end]) > 0.7 
    @test all(isreal.(model.P))
    @test all(model.P .>= 0)
    @test all(model.P .<= 1)
    @test isapprox(sum(model.P, dims=1), ones(1,3))
    @test all(expected_duration(model) .> 0)
    @test size(filtered_probs(model)) == (T, k)
    @test size(smoothed_probs(model)) == (T, k)
    @test model.nlopt_msg == :XTOL_REACHED

end

@testset "stochastic component - non-switching variance" begin
    k = 3
    μ = [1.0, -0.5, 0.12] 
    σ = [0.2, 0.2, 0.2] 
    P = [0.9 0.05 0.1; 0.05 0.85 0.05; 0.05 0.1 0.85]
    T = 500

    y, s_t, X = generate_msm(μ, σ, P, T)

    model = MSModel(y, k, switching_var = false,
                            random_search = n_rnd_search)
                
    @test all(model.σ .== model.σ[1])
    @test abs(model.σ[1] .- σ[1]) < 0.2
    @test model.nlopt_msg == :XTOL_REACHED
end

@testset "stochastic component - tvtp" begin
    
    k = 2
    μ = [1.2, -0.1] 
    σ = [0.3,  0.2] 
    P = [0.5 0.05
        0.5 0.95]
    #P = [0.9 0.05 0.1; 0.05 0.85 0.05; 0.05 0.1 0.85]
    #δ = [4.0, 0.1, 0.2, 0.1, 6, 0.25, -0.4, 0.1]
    δ = [2.2, 0.9]
    T = 500

    y, s_t, X = generate_msm(μ, σ, P, T, δ = δ, tvtp_intercept = false)
    x_tvtp = reshape(X[:,2], T, 1)

    model = MSModel(y, k, intercept = "switching", 
                            exog_tvtp = x_tvtp, 
                            maxtime = 100,
                            random_search = n_rnd_search)

    @test get_std_errors(model) isa Vector{Float64}                                
    @test model.nlopt_msg == :XTOL_REACHED
    @test isnothing(coeftable_tvtp(model))
    @test size(expected_duration(model)) == (T, k)
    @test abs(cor([[MarSwitching.P_tvtp(x_tvtp[i], δ, k, 1)[2] for i in 1:T] [MarSwitching.P_tvtp(x_tvtp[i], model.δ, k, 1)[2] for i in 1:T]])[2]) > 0.8
    @test MarSwitching.loglik_tvtp(model.raw_params, model.x, k, model.n_β, model.n_β_ns, model.intercept, model.switching_var, 1)[1] == model.Likelihood
    ξ_t = filtered_probs(model)

    @test all(abs.(sort([model.β[i][1] for i in 1:model.k]) .- sort(μ)) .< 0.3)
    @test maximum(cor([ξ_t s_t])[1:2, end]) > 0.6

    ŷ, P̂ = MarSwitching.predict(model, true)

    @test (cor([ŷ y])[2]) > 0.6

    ŷ, P̂ = MarSwitching.predict(model, false)

    @test (cor([ŷ y[1:end-1]])[2]) > 0.6
end

@testset "predict function" begin 
    k = 3
    μ = [1.0, -0.5, 0.0] 
    β_ns = Vector{Float64}([0.7])
    σ = [0.3,  0.6, 0.8] 
    P = [0.9 0.05 0.1; 0.05 0.85 0.05; 0.05 0.1 0.85]
    T = 1000

    y, s_t, X = generate_msm(μ, σ, P, T, β_ns = β_ns)

    model = MSModel(y, k, intercept = "switching", 
                            exog_vars = reshape(X[:,2],T,1),
                            random_search = n_rnd_search)
                
    my_mean(x) = sum(x) / length(x)
    my_std(x) = sqrt(sum((x .- my_mean(x)).^2) / (length(x)-1))
    corr(x,y) = sum((x .- my_mean(x)) .* (y .- my_mean(y))) / (my_std(x)*my_std(y)*(length(x)-1))
    
    y_pred, ξ_t = MarSwitching.predict(model, true)

    @test corr(y_pred, y)[1] > 0.7

    T_pred = 400
    y_oos, _, X_oos = generate_msm(μ, σ, P, T_pred, β_ns = β_ns)

    y_pred2, ξ_t2 = MarSwitching.predict(model, true, y = y_oos, exog_vars = reshape(X_oos[:,2],T_pred,1))

    @test corr(y_pred2, y_oos)[1] > 0.6

    y_pred3, ξ_t3 = MarSwitching.predict(model, false, y = y_oos, exog_vars = reshape(X_oos[:,2],T_pred,1))

    @test corr(y_pred3, y_oos[2:end])[1] > 0.5
    @test model.nlopt_msg == :XTOL_REACHED

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
                    σ, β, P = MarSwitching.trans_θ(θ, k_i, n_β_i, n_β_ns_i, int, true, false)
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

    k = 3
    β = Vector{Float64}([-1.5, 0.9, 0.0])
    β_ns = Vector{Float64}([-0.33])
    σ = [0.7,  0.3, 0.1] 
    P = [0.9 0.05 0.1; 0.05 0.85 0.05; 0.05 0.1 0.85]

    θ = [σ; β; β_ns; vec(P[2:end, :])]

    σ_, β_ = MarSwitching.vec2param_nointercept(θ, k, 1, 1, true)

    @test all([β_[i][1] == 0 for i in 1:k])
    @test σ_ == σ

end

@testset "Less crucial functions" begin
    @test add_lags([1.0,2.0,3.0,4.0], 1) == [2.0 1.0; 3.0 2.0; 4.0 3.0]
    A = rand(4,4)
    @test all(abs.(MarSwitching.mp_inverse(A) .- pinv(A)) .< 0.001)
end



