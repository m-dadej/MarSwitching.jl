using MarSwitching
using Test
using StatsBase
using LinearAlgebra
using Random

seed = rand(1:100)
println("Seed: ", seed)
Random.seed!(seed)

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
    @test ergodic_probs(model.P) isa Vector{Float64}

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
                            exog_switching_vars = X[:,2],
                            random_search_em = 3)

    @test model.nlopt_msg == :XTOL_REACHED                    
    @test all(abs.(sort([model.β[i][1] for i in 1:model.k]) .- sort(μ)) .< 0.3)
    @test all(abs.(sort([model.β[i][2] for i in 1:model.k]) .- sort(β)) .< 0.3)
    @test maximum(cor([filtered_probs(model) (s_t .== 3)])[1:3,end]) > 0.6 
    @test maximum(cor([smoothed_probs(model) (s_t .== 3)])[1:3,end]) > 0.7 

    y_, s_t_, X_ = generate_msm(model, 1000)

    model_ = MSModel(y_, k, intercept = "switching", 
                            exog_switching_vars = X_[:,2],
                            random_search_em = 4)

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
                            exog_vars = X[:,2],
                            random_search_em = 3) 

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
                            exog_vars = X[:,2],
                            random_search_em = 3)

                          
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
                            exog_vars = X[:,3],
                            random_search_em = 3)

    @test maximum(cor([filtered_probs(model) (s_t .== 3)])[1:3,end]) > 0.5 
    @test maximum(cor([smoothed_probs(model) (s_t .== 3)])[1:3,end]) > 0.5 

    @test all([model.β[i][1] == 0 for i in 1:model.k])
    @test model.nlopt_msg == :XTOL_REACHED
end

@testset "stochastic component - 3 state model every exogenous vars" begin

    k = 3
    μ = [1.0, -0.5, 0.12] 
    β = Vector{Float64}([-1.5, 0.9, 0.0, 0.6, -1.8, 0.45]) # S_1 = {-1.5, 0.9}, S_2 = {0.0., 0.6}...
    β_ns = Vector{Float64}([0.3333])
    σ = [0.4,  0.5, 0.2] 
    #P = [0.7 0.2; 0.3 0.8]
    P = [0.9 0.05 0.1; 0.05 0.85 0.05; 0.05 0.1 0.85]
    T = 1000

    y, s_t, X = generate_msm(μ, σ, P, T, β = β, β_ns = β_ns)

    model = MSModel(y, k, intercept = "switching", 
                            exog_switching_vars = X[:,2:3],
                            exog_vars = X[:,4],
                            random_search_em = 3,
                            random_search = 2)

    @test all(abs.(sort([model.β[i][1] for i in 1:model.k]) .- sort(μ)) .< 0.2)
    @test all(abs.(sort([model.β[i][2] for i in 1:model.k]) .- sort(β[1:2:k*2])) .< 0.2)
    @test all(abs.(sort([model.β[i][3] for i in 1:model.k]) .- sort(β[2:2:k*2])) .< 0.2)


    # to add tests below we need better x0 for P or random search 
    # because around eery 2 estimations the P is very biased                            
    @test maximum(cor([filtered_probs(model) (s_t .== 3)])[1:3,end]) > 0.6 
    @test maximum(cor([smoothed_probs(model) (s_t .== 3)])[1:3,end]) > 0.7 
    @test all(isreal.(model.P))
    @test all(model.P .>= 0)
    @test all(model.P .<= 1)
    @test isapprox(sum(model.P, dims=1), ones(1,3))
    @test all(expected_duration(model) .> 0)
    @test size(filtered_probs(model)) == (T, k)
    @test size(smoothed_probs(model)) == (T, k)
    @test model.nlopt_msg == :XTOL_REACHED

    @test isapprox(sum(ergodic_probs(model.P)), 1.0)
end

@testset "stochastic component - non-switching variance" begin
    k = 3
    μ = [1.0, -0.5, 0.12] 
    σ = [0.2, 0.2, 0.2] 
    P = [0.9 0.05 0.1; 0.05 0.85 0.05; 0.05 0.1 0.85]
    T = 500

    y, s_t, X = generate_msm(μ, σ, P, T)

    model = MSModel(y, k, switching_var = false, random_search_em=3)
                
    @test all(model.σ .== model.σ[1])
    @test abs(model.σ[1] .- σ[1]) < 0.2
    @test model.nlopt_msg == :XTOL_REACHED
end

@testset "stochastic component - model selection" begin
    k = 3
    μ = [1.0, -0.5, 0.12] 
    β_ns = Vector{Float64}([0.633])
    σ = [1.7,  0.8, 0.9] 
    P = [0.9 0.05 0.1; 0.05 0.85 0.05; 0.05 0.1 0.85]
    T = 200

    y, s_t, X = generate_msm(μ, σ, P, T, β_ns = β_ns)

    model, crit, params = grid_search_msm(y, X[:,2], 
                                        random_search_em = 2,
                                        random_n = 3);
    @test model isa MSM
end

@testset "stochastic component - tvtp" begin
    
    k = 2
    μ = [1.2, -0.1] 
    σ = [0.2,  0.1] 
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
                            random_search_em = 5,
                            random_search = 5)

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

    @test all(isapprox.(sum(ergodic_probs(model), dims=2), 1.0))
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
                            random_search_em = 3)
                
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
                    σ, β, P, ν = MarSwitching.trans_θ(θ, k_i, n_β_i, n_β_ns_i, int, true, false)
                    println("k: $k_i, n_β: $n_β_i, n_β_ns: $n_β_ns_i, intercept: $int")

                    @test size(σ)[1] == k_i
                    @test size(σ)[1] == k_i
                    @test size(β)[1] == k_i
                    @test size(β[1])[1] == n_β_i+ 1 + n_β_ns_i
                    @test size(P)[1] == k_i
                    @test isempty(ν)

                    # Student-t parameterization includes k log(ν) parameters after σ
                    θ_t = [rand(k_i); fill(log(5.0), k_i); rand(Uniform(-5, 5), n_int); rand(Uniform(-5, 5), n_β_i*k_i); rand(Uniform(-5, 5), n_β_ns_i); rand(k_i*(k_i-1))]
                    σ_t, β_t, P_t, ν_t = MarSwitching.trans_θ(θ_t, k_i, n_β_i, n_β_ns_i, int, true, false, :t)
                    @test length(ν_t) == k_i
                    @test all(ν_t .> 0.0)
                    @test all(isapprox.(ν_t, 5.0; rtol=1e-10))
                    @test size(σ_t)[1] == k_i
                    @test size(P_t)[1] == k_i

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

@testset "Student-t error distribution" begin
    k = 2
    μ = [1.0, -0.5]
    β = [-1.5, 0.0]
    σ = [1.1, 0.8]
    ν = [1.0, 4.0]
    P = [0.9 0.05; 0.1 0.95]
    T = 2500

    y, s_t, X = generate_msm(μ, σ, P, T, β = β, error_dist = :t, ν = ν)

    model = MSModel(y, k, intercept = "switching",
                    error_dist = :t,
                    exog_switching_vars = X[:,2],
                    random_search_em = 6,
                    random_search = 6,
                    verbose = false)

    @test model.error_dist == :t
    @test length(model.ν) == k
    @test all(model.ν .> 0.0)
    @test !isnan(model.Likelihood) && (model.Likelihood != Inf)
    @test MarSwitching.loglik(model.raw_params, model.x, k, model.n_β, model.n_β_ns,
                              model.intercept, model.switching_var, :t)[1] == model.Likelihood
    @test size(filtered_probs(model)) == (T, k)
    @test size(smoothed_probs(model)) == (T, k)
    @test size(get_std_errors(model))[1] == size(model.raw_params)[1]
    @test isnothing(summary_msm(model))

    # match states by sorted intercept (label switching)
    ord_t = sortperm(μ)
    ord_e = sortperm([model.β[i][1] for i in 1:k])
    @test all(abs.([model.β[i][1] for i in ord_e] .- μ[ord_t]) .< 0.25)
    @test all(abs.([model.β[i][2] for i in ord_e] .- β[ord_t]) .< 0.25)
    @test all(abs.(model.σ[ord_e] .- σ[ord_t]) .< 0.25)
    @test all(abs.(model.ν[ord_e] .- ν[ord_t]) ./ ν[ord_t] .< 0.6)

    y2, s2, X2 = generate_msm(model, 500)
    @test length(y2) == 500
    @test sort(unique(s2)) == collect(1:k)

    @test_throws AssertionError generate_msm(μ, σ, P, 100, error_dist = :t, ν = [-1.0, 5.0])
end

@testset "GED error distribution" begin
    # GED with ν=2 must match Gaussian density
    y = collect(-2.0:0.5:2.0)
    μ = zeros(length(y))
    d_n = MarSwitching.error_density(:normal, y, μ, 1.0)
    d_g = MarSwitching.error_density(:ged, y, μ, 1.0, 2.0)
    @test all(isapprox.(d_n, d_g; rtol=1e-8))

    k = 2
    μ = [1.0, -0.5]
    β = [-1.5, 0.0]
    σ = [1.0, 0.7]
    ν = [1.2, 2.5]   # heavier than normal / thinner than normal
    P = [0.9 0.05; 0.1 0.95]
    T = 1200

    y, s_t, X = generate_msm(μ, σ, P, T, β = β, error_dist = :ged, ν = ν)

    model = MSModel(y, k, intercept = "switching",
                    error_dist = :ged,
                    exog_switching_vars = X[:,2],
                    random_search_em = 3,
                    random_search = 3,
                    verbose = false)

    @test model.error_dist == :ged
    @test length(model.ν) == k
    @test all(model.ν .> 0.0)
    @test !isnan(model.Likelihood) && (model.Likelihood != Inf)
    @test MarSwitching.loglik(model.raw_params, model.x, k, model.n_β, model.n_β_ns,
                              model.intercept, model.switching_var, :ged)[1] == model.Likelihood
    @test size(filtered_probs(model)) == (T, k)
    @test size(get_std_errors(model))[1] == size(model.raw_params)[1]
    @test isnothing(summary_msm(model))

    ord_t = sortperm(μ)
    ord_e = sortperm([model.β[i][1] for i in 1:k])
    @test all(abs.([model.β[i][1] for i in ord_e] .- μ[ord_t]) .< 0.3)
    @test all(abs.([model.β[i][2] for i in ord_e] .- β[ord_t]) .< 0.3)
    @test all(abs.(model.σ[ord_e] .- σ[ord_t]) .< 0.3)

    y2, s2, _ = generate_msm(model, 400)
    @test length(y2) == 400
    @test sort(unique(s2)) == collect(1:k)
end

@testset "MS-ARCH: arch_var" begin
    y  = [1.0, 2.0, -1.0, 0.5]
    μ  = [zeros(4), zeros(4)]
    h  = MarSwitching.arch_var(y, μ, [0.5, 1.0], [[0.4], [0.0]], 2, 1)
    e2 = y.^2
    b  = sum(e2) / 4

    @test h[1,1] ≈ 0.5 + 0.4*b        # pre-sample backcast
    @test h[2,1] ≈ 0.5 + 0.4*e2[1]
    @test h[4,1] ≈ 0.5 + 0.4*e2[3]
    @test all(h[:,2] .≈ 1.0)          # α = 0 -> constant variance
    @test size(h) == (4, 2)
end

@testset "MS-ARCH: trans_θ_arch parameter transformation" begin
    k, q, n_β, n_β_ns = 3, 2, 1, 1
    ω_raw = [0.5, 1.0, 1.5]
    α_raw = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]   # state-major: s1=[.1,.2], s2=[.3,.4], s3=[.5,.6]
    rest  = [randn(k); randn(n_β*k); randn(n_β_ns); rand(k*(k-1))]

    ω, α, β, P, ν = MarSwitching.trans_θ_arch([ω_raw; α_raw; rest], k, n_β, n_β_ns,
                                              "switching", true, false, :normal, q)

    @test ω ≈ ω_raw.^2
    @test α[1] ≈ [0.01, 0.04]
    @test α[3] ≈ [0.25, 0.36]
    @test length(β) == k && length(β[1]) == n_β + n_β_ns + 1
    @test size(P) == (k, k)
    @test isempty(ν)

    # β and P blocks must parse identically to the constant-variance model on the same tail
    _, β2, P2, _ = MarSwitching.trans_θ([ω_raw; rest], k, n_β, n_β_ns, "switching", true, false)
    @test β == β2
    @test P == P2

    # shared ω / α under switching_var = false
    ωn, αn, _, _, _ = MarSwitching.trans_θ_arch([[0.5]; [0.1, 0.2]; rest], k, n_β, n_β_ns,
                                                "switching", false, false, :normal, q)
    @test all(ωn .== 0.25)
    @test all(αn[s] ≈ [0.01, 0.04] for s in 1:k)

    # regression: α sits AFTER ν in the raw layout so that the ν noise/multi-start logic
    # in MSModel() doesn't collide with it. Check ν is still read from the right offset
    # (and unaffected by the presence of the α block), and α is unaffected by ν's presence.
    θ_t = [ω_raw; fill(log(6.0), k); α_raw; rest]
    _, αt, _, _, νt = MarSwitching.trans_θ_arch(θ_t, k, n_β, n_β_ns, "switching", true, false, :t, q)
    @test all(isapprox.(νt, 6.0; rtol=1e-10))
    @test αt[1] ≈ [0.01, 0.04]
end

@testset "MS-ARCH nests the constant-variance model at α = 0" begin
    k, T = 2, 400
    y, _, _ = generate_msm([1.0, -0.5], [0.5, 1.2], [0.9 0.1; 0.1 0.9], T)
    x     = [y ones(T)]
    σ_raw = [0.7, 1.1]
    tail  = [1.0, -0.5, 0.6, 0.6]   # intercepts; P raw

    ll0 = MarSwitching.loglik([σ_raw; tail], x, k, 0, 0, "switching", true)[1]
    lla = MarSwitching.loglik([σ_raw.^2; zeros(k); tail], x, k, 0, 0,
                              "switching", true, :normal, 1)[1]
    @test isapprox(ll0, lla; rtol = 1e-10)
end

@testset "MSARCHModel estimation" begin
    k, T = 2, 1200
    μ = [0.0, 0.0]                 # regimes identified through variance only
    ω = [0.10, 1.00]
    α = [[0.25], [0.55]]
    P = [0.97 0.03; 0.03 0.97]
    y, s_t, X = generate_msm(μ, ω, α, P, T)

    model = MSARCHModel(y, k, 1, intercept = "no", random_search_em = 2,
                        maxtime = 45, verbose = false)

    @test model isa MSM
    @test model.q == 1
    @test length(model.ω) == k && length(model.α) == k && length(model.α[1]) == 1
    @test all(model.ω .> 0) && all(model.α[s][1] >= 0 for s in 1:k)
    @test !isnan(model.Likelihood) && (model.Likelihood != Inf)

    @test MarSwitching.loglik(model.raw_params, model.x, k, model.n_β, model.n_β_ns,
                              model.intercept, model.switching_var, :normal, model.q)[1] == model.Likelihood

    @test size(filtered_probs(model)) == (T, k)
    @test size(smoothed_probs(model)) == (T, k)
    @test size(get_std_errors(model))[1] == size(model.raw_params)[1]
    @test isnothing(summary_msm(model))
    @test isnothing(display(model))

    # low-vol vs high-vol regime should be identifiable even under a fresh random seed
    ord = sortperm(model.ω)
    @test model.ω[ord][1] < model.ω[ord][2]
    @test maximum(cor([smoothed_probs(model) (s_t .== 2)])[1:k, end]) > 0.4

    # conditional_variance(): right shape, positive, and its time-average matches
    # model.σ.^2 (σ is defined as sqrt(mean(h)), see MSModel())
    cv = conditional_variance(model)
    @test size(cv) == (T, k)
    @test all(cv .> 0)
    @test all(isapprox.([sum(view(cv, :, s))/T for s in 1:k], model.σ.^2; rtol = 1e-8))

    # an ARCH-aware model should not fit ARCH-generated data worse than a constant-variance one
    m0 = MSModel(y, k, intercept = "no", random_search_em = 2, maxtime = 20, verbose = false)
    @test model.Likelihood > m0.Likelihood - 2.0

    y2, s2, _ = generate_msm(model, 500)
    @test length(y2) == 500 && sort(unique(s2)) == collect(1:k)

    @test_throws AssertionError generate_msm(μ, ω, [[0.2], [0.3, 0.1]], P, 100)
    @test_throws AssertionError MSARCHModel(y, k, 0)
end

@testset "MS-ARCH: switching_var = false and q = 0 regression" begin
    k, T = 2, 800
    y, _, _ = generate_msm([0.0, 0.0], [0.3, 0.3], [0.9 0.1; 0.1 0.9], T)

    m = MSARCHModel(y, k, 1, switching_var = false, intercept = "no", maxtime = 20, verbose = false)
    @test all(m.ω .== m.ω[1])
    @test all(m.α[s] == m.α[1] for s in 1:k)
    @test length(m.raw_params) == 4   # ω(1) + α(1) + P raw(2); no ν, no β, no intercept

    m0 = MSModel(y, k, intercept = "no", verbose = false)
    @test m0.q == 0
    @test isempty(m0.ω)
    @test isempty(m0.α)
end

@testset "MS-ARCH with TVTP" begin
    k, T = 2, 400
    μ = [0.0, 0.0]
    ω = [0.15, 0.9]
    α = [[0.2], [0.4]]
    P = [0.9 0.1; 0.1 0.9]
    δ = [2.0, 0.5]

    y, s_t, X = generate_msm(μ, ω, α, P, T, δ = δ, tvtp_intercept = false)
    x_tvtp = reshape(X[:,2], T, 1)

    model = MSARCHModel(y, k, 1, intercept = "no", exog_tvtp = x_tvtp,
                        maxtime = 30, random_search_em = 1, verbose = false)

    @test model isa MSM
    @test model.q == 1
    @test isempty(model.P)
    @test !isempty(model.δ)
    @test size(filtered_probs(model)) == (T, k)
    @test size(get_std_errors(model))[1] == size(model.raw_params)[1]
    @test isnothing(summary_msm(model))
    @test MarSwitching.loglik_tvtp(model.raw_params, model.x, k, model.n_β, model.n_β_ns,
                                   model.intercept, model.switching_var, 1, :normal, model.q)[1] == model.Likelihood
end

@testset "MS-GARCH: garch_var" begin
    y  = [1.0, 2.0, -1.0, 0.5]
    μ  = [zeros(4), zeros(4)]
    h  = MarSwitching.garch_var(y, μ, [0.5, 1.0], [[0.2], [0.3]], [[0.6], [0.0]], 2, 1, 1)
    e2 = y.^2
    b  = sum(e2) / 4

    @test h[1,1] ≈ 0.5 + 0.2*b + 0.6*b              # pre-sample backcast for both ε² and h
    @test h[2,1] ≈ 0.5 + 0.2*e2[1] + 0.6*h[1,1]
    @test h[4,1] ≈ 0.5 + 0.2*e2[3] + 0.6*h[3,1]
    @test size(h) == (4, 2)
    # γ = 0 reproduces arch_var() exactly
    @test h[:,2] == MarSwitching.arch_var(y, μ, [0.5, 1.0], [[0.2], [0.3]], 2, 1)[:,2]
end

@testset "MS-GARCH: trans_θ_garch parameter transformation" begin
    k, q, p, n_β, n_β_ns = 3, 2, 1, 1, 1
    ω_raw = [0.5, 1.0, 1.5]
    α_raw = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]   # state-major: s1=[.1,.2], s2=[.3,.4], s3=[.5,.6]
    γ_raw = [2.0, 1.0, 3.0]                  # s1, s2, s3
    rest  = [randn(k); randn(n_β*k); randn(n_β_ns); rand(k*(k-1))]

    ω, α, γ, β, P, ν = MarSwitching.trans_θ_garch([ω_raw; α_raw; γ_raw; rest], k, n_β, n_β_ns,
                                                  "switching", true, false, :normal, q, p)
    D1 = 1 + 0.1^2 + 0.2^2 + 2.0^2
    D3 = 1 + 0.5^2 + 0.6^2 + 3.0^2
    @test ω ≈ ω_raw.^2
    @test α[1] ≈ [0.01, 0.04] ./ D1
    @test γ[1] ≈ [4.0] ./ D1
    @test α[3] ≈ [0.25, 0.36] ./ D3
    @test γ[3] ≈ [9.0] ./ D3
    @test all(sum(α[s]) + sum(γ[s]) < 1 for s in 1:k)    # stationarity by construction
    @test isempty(ν)

    # β and P blocks must parse identically to the constant-variance model on the same tail
    _, β2, P2, _ = MarSwitching.trans_θ([ω_raw; rest], k, n_β, n_β_ns, "switching", true, false)
    @test β == β2
    @test P == P2

    # garch_raw() inverts garch_coefs()
    c = [0.1, 0.05, 0.8]
    @test MarSwitching.garch_coefs(MarSwitching.garch_raw(c)) ≈ c
    @test_throws AssertionError MarSwitching.garch_raw([0.5, 0.6])

    # shared ω / α / γ under switching_var = false
    ωn, αn, γn, _, _, _ = MarSwitching.trans_θ_garch([[0.5]; [0.1, 0.2]; [2.0]; rest], k, n_β, n_β_ns,
                                                     "switching", false, false, :normal, q, p)
    @test all(ωn .== 0.25)
    @test all(αn[s] ≈ [0.01, 0.04] ./ D1 for s in 1:k)
    @test all(γn[s] ≈ [4.0] ./ D1 for s in 1:k)

    # ν still read from the right offset with both α and γ blocks present
    θ_t = [ω_raw; fill(log(6.0), k); α_raw; γ_raw; rest]
    _, αt, γt, _, _, νt = MarSwitching.trans_θ_garch(θ_t, k, n_β, n_β_ns, "switching", true, false, :t, q, p)
    @test all(isapprox.(νt, 6.0; rtol=1e-10))
    @test αt[1] ≈ α[1] && γt[1] ≈ γ[1]

    # TVTP variant returns no P
    @test length(MarSwitching.trans_θ_garch([ω_raw; α_raw; γ_raw; rest], k, n_β, n_β_ns,
                                            "switching", true, true, :normal, q, p)) == 5
end

@testset "MS-GARCH nests MS-ARCH at γ = 0" begin
    k, T = 2, 400
    y, _, _ = generate_msm([0.0, 0.0], [0.2, 0.8], [[0.3], [0.1]], [0.9 0.1; 0.1 0.9], T)
    x     = [y zeros(T)]
    ω_raw = [0.4, 0.9]
    a     = [0.6, 0.3]                       # GARCH-path α raw
    P_raw = [0.9/0.1, 0.1/0.9]
    α_c   = a.^2 ./ (1 .+ a.^2)              # α implied by garch_coefs() when γ raw = 0

    ll_g = MarSwitching.loglik([ω_raw; a; zeros(k); P_raw], x, k, 0, 0, "no", true, :normal, 1, 1)[1]
    ll_a = MarSwitching.loglik([ω_raw; sqrt.(α_c); P_raw], x, k, 0, 0, "no", true, :normal, 1)[1]
    @test isapprox(ll_g, ll_a; rtol = 1e-10)

    # same on the TVTP path (constant tvtp regressor)
    xt = [x ones(T)]
    δ  = [2.0, -2.0]
    ll_gt = MarSwitching.loglik_tvtp([ω_raw; a; zeros(k); δ], xt, k, 0, 0, "no", true, 1, :normal, 1, 1)[1]
    ll_at = MarSwitching.loglik_tvtp([ω_raw; sqrt.(α_c); δ], xt, k, 0, 0, "no", true, 1, :normal, 1)[1]
    @test isapprox(ll_gt, ll_at; rtol = 1e-10)

    # MS-ARCH simulation is MS-GARCH simulation with γ = 0: identical draws for the same seed
    μ, ω, α, P = [0.0, 0.5], [0.2, 0.8], [[0.3], [0.1]], [0.9 0.1; 0.1 0.9]
    sd = rand(UInt32)
    Random.seed!(sd); y1, s1, X1 = generate_msm(μ, ω, α, P, 300)
    Random.seed!(sd); y2, s2, X2 = generate_msm(μ, ω, α, [[0.0], [0.0]], P, 300)
    @test y1 == y2 && s1 == s2 && X1 == X2
end

@testset "MSGARCHModel estimation" begin
    k, T = 2, 1500
    μ = [0.0, 0.0]                 # regimes identified through variance only
    ω = [0.05, 0.60]
    α = [[0.05], [0.15]]
    γ = [[0.85], [0.60]]
    P = [0.98 0.02; 0.02 0.98]
    y, s_t, X = generate_msm(μ, ω, α, γ, P, T)

    model = MSGARCHModel(y, k, 1, 1, intercept = "no", random_search_em = 2,
                         maxtime = 30, verbose = false)

    @test model isa MSM
    @test model.p == 1 && model.q == 1
    @test length(model.γ) == k && length(model.γ[1]) == 1
    @test length(model.raw_params) == 8      # ω(2) + α(2) + γ(2) + P raw(2)
    @test all(model.ω .> 0)
    @test all(model.α[s][1] >= 0 && model.γ[s][1] >= 0 for s in 1:k)
    @test all(sum(model.α[s]) + sum(model.γ[s]) < 1 for s in 1:k)
    @test !isnan(model.Likelihood) && (model.Likelihood != Inf)

    @test MarSwitching.loglik(model.raw_params, model.x, k, model.n_β, model.n_β_ns,
                              model.intercept, model.switching_var, :normal, model.q, model.p)[1] == model.Likelihood

    # the fit should be at least about as good as the data-generating parameters
    r_true = [MarSwitching.garch_raw([α[s]; γ[s]]) for s in 1:k]
    θ_true = [sqrt.(ω); r_true[1][1]; r_true[2][1]; r_true[1][2]; r_true[2][2];
              P[1,1]/(1-P[1,1]); P[1,2]/(1-P[1,2])]
    ll_true = MarSwitching.loglik(θ_true, model.x, k, 0, 0, "no", true, :normal, 1, 1)[1]
    @test model.Likelihood > ll_true - 15.0

    @test size(filtered_probs(model)) == (T, k)
    @test size(smoothed_probs(model)) == (T, k)
    @test size(get_std_errors(model))[1] == size(model.raw_params)[1]
    @test isnothing(summary_msm(model))
    @test isnothing(display(model))

    @test maximum(cor([smoothed_probs(model) (s_t .== 2)])[1:k, end]) > 0.4

    cv = conditional_variance(model)
    @test size(cv) == (T, k)
    @test all(cv .> 0)
    @test all(isapprox.([sum(view(cv, :, s))/T for s in 1:k], model.σ.^2; rtol = 1e-8))

    # a GARCH model should not fit GARCH-generated data worse than an ARCH one
    m_arch = MSARCHModel(y, k, 1, intercept = "no", random_search_em = 2, maxtime = 20, verbose = false)
    @test model.Likelihood > m_arch.Likelihood - 2.0

    y2, s2, _ = generate_msm(model, 500)
    @test length(y2) == 500 && sort(unique(s2)) == collect(1:k)

    @test_throws AssertionError generate_msm(μ, ω, α, [[0.5], [0.3, 0.1]], P, 100)
    @test_throws AssertionError generate_msm(μ, ω, α, [[-0.1], [0.3]], P, 100)
    @test_throws AssertionError MSGARCHModel(y, k, 0, 1)
    @test_throws AssertionError MSGARCHModel(y, k, 1, 0)
    @test_throws AssertionError MSModel(y, k, p = 1)        # GARCH terms without ARCH terms
end

@testset "MS-GARCH: switching_var = false and p = 0 regression" begin
    k, T = 2, 800
    y, _, _ = generate_msm([0.0, 0.0], [0.1, 0.1], [[0.1], [0.1]], [[0.8], [0.8]], [0.9 0.1; 0.1 0.9], T)

    m = MSGARCHModel(y, k, 1, 1, switching_var = false, intercept = "no", maxtime = 15, verbose = false)
    @test all(m.ω .== m.ω[1])
    @test all(m.α[s] == m.α[1] for s in 1:k)
    @test all(m.γ[s] == m.γ[1] for s in 1:k)
    @test length(m.raw_params) == 5   # ω(1) + α(1) + γ(1) + P raw(2)
    @test isnothing(summary_msm(m))

    m_a = MSARCHModel(y, k, 1, intercept = "no", maxtime = 15, verbose = false)
    @test m_a.p == 0 && isempty(m_a.γ)
    m0 = MSModel(y, k, intercept = "no", verbose = false)
    @test m0.p == 0 && isempty(m0.γ)
end

@testset "MS-GARCH with TVTP" begin
    k, T = 2, 600
    μ = [0.0, 0.0]
    ω = [0.05, 0.5]
    α = [[0.1], [0.2]]
    γ = [[0.8], [0.5]]
    P = [0.9 0.1; 0.1 0.9]
    δ = [2.0, 0.5]

    y, s_t, X = generate_msm(μ, ω, α, γ, P, T, δ = δ, tvtp_intercept = false)
    x_tvtp = reshape(X[:,2], T, 1)

    # short maxtime: this testset checks the TVTP plumbing, not estimation quality
    model = MSGARCHModel(y, k, 1, 1, intercept = "no", exog_tvtp = x_tvtp,
                         maxtime = 6, random_search_em = 1, verbose = false)

    @test model isa MSM
    @test model.p == 1
    @test isempty(model.P)
    @test !isempty(model.δ)
    @test size(filtered_probs(model)) == (T, k)
    @test size(conditional_variance(model)) == (T, k)
    @test size(get_std_errors(model))[1] == size(model.raw_params)[1]
    @test isnothing(summary_msm(model))
    @test MarSwitching.loglik_tvtp(model.raw_params, model.x, k, model.n_β, model.n_β_ns,
                                   model.intercept, model.switching_var, 1, :normal, model.q, model.p)[1] == model.Likelihood
end

@testset "MS-GARCH with exogenous mean variables and Student-t errors" begin
    k, T = 2, 800
    μ    = [0.5, -0.5]
    β    = [1.0, -1.0]                # switching slope of one exogenous variable
    β_ns = [0.3]                      # non-switching slope of another
    ω    = [0.05, 0.4]
    α    = [[0.1], [0.15]]
    γ    = [[0.8], [0.6]]
    P    = [0.95 0.05; 0.05 0.95]

    y, s_t, X = generate_msm(μ, ω, α, γ, P, T, β = β, β_ns = β_ns, error_dist = :t, ν = [6.0, 6.0])

    model = MSGARCHModel(y, k, 1, 1, exog_switching_vars = X[:, 2], exog_vars = X[:, 3],
                         error_dist = :t, maxtime = 20, random_search_em = 2, verbose = false)

    @test model.p == 1 && model.n_β == 1 && model.n_β_ns == 1
    @test length(model.ν) == k && all(model.ν .> 0)
    # ω(2) + ν(2) + α(2) + γ(2) + intercepts(2) + switching β(2) + non-switching β(1) + P(2)
    @test length(model.raw_params) == 15
    @test MarSwitching.loglik(model.raw_params, model.x, k, model.n_β, model.n_β_ns,
                              model.intercept, model.switching_var, :t, model.q, model.p)[1] == model.Likelihood
    # switching slopes should be recovered with the right signs (regime order is arbitrary)
    @test sort([model.β[s][2] for s in 1:k]) ≈ [-1.0, 1.0] atol = 0.3
    @test model.β[1][3] ≈ 0.3 atol = 0.15
    @test isnothing(summary_msm(model))
    @test size(MarSwitching.predict(model)[1], 1) == T - 1
end

@testset "Hamilton filter matches the original per-step loop" begin
    # the loop hamilton_filter() replaced, kept here as the reference
    function filter_reference(η, P, ξ_0, logsum)
        T, k = size(η); ξ = zeros(T, k); L = zeros(T); ξ_next = zeros(k)
        for t in 1:T
            ξ[t,:] = t == 1 ? ξ_0 : view(ξ, t-1, :)
            mul!(ξ_next, P, view(ξ, t, :))
            L[t] = max(view(η, t, :)'ξ_next, 1e-300)
            @views @. ξ[t,:] = (1/L[t]) * ξ_next * η[t, :]
        end
        return (logsum ? sum(log.(L)) : L), ξ
    end
    for k in 2:4
        P = rand(k, k) .+ 5 .* Matrix(1.0I, k, k); P = P ./ sum(P, dims = 1)
        η = rand(500, k) .* 3
        η[17, :] .= 0.0                                   # exercises the 1e-300 floor
        ξ_0 = ergodic_probs(P, k)
        for logsum in (true, false)
            a = MarSwitching.hamilton_filter(η, P, ξ_0, logsum)
            b = filter_reference(η, P, ξ_0, logsum)
            @test all(isapprox.(a[1], b[1]; rtol = 1e-12))
            @test isapprox(a[2], b[2]; rtol = 1e-12)
        end
    end
end

@testset "MS-GARCH optimizer safeguards" begin
    # bounded_gradient!: never evaluates f outside [lb, ub], central differences inside
    lb, ub = [0.0, -Inf, -Inf], [Inf, 1.0, Inf]
    f(θ) = any(θ .< lb) || any(θ .> ub) ? error("evaluated outside the bounds") : θ[1]^2 + θ[1] + 2θ[2]^2 + sin(θ[3])
    g = zeros(3)
    @test MarSwitching.bounded_gradient!(g, f, [0.0, 1.0, 0.3], lb, ub) == f([0.0, 1.0, 0.3])
    @test g ≈ [1.0, 4.0, cos(0.3)] rtol = 1e-6          # one-sided at θ[1] = lb, θ[2] = ub
    MarSwitching.bounded_gradient!(g, f, [0.7, -0.4, 0.3], lb, ub)
    @test g ≈ [2*0.7 + 1, -1.6, cos(0.3)] rtol = 1e-8   # interior: central differences

    # persistent_P_raw: P[s,s] = pstay, the rest spread evenly, in trans_θ()'s layout
    for k in 2:4
        _, _, P, _ = MarSwitching.trans_θ([ones(k); zeros(k); MarSwitching.persistent_P_raw(k, 0.95)],
                                          k, 0, 0, "switching", true, false)
        @test all(diag(P) .≈ 0.95)
        @test all(P[i, j] ≈ 0.05 / (k - 1) for i in 1:k, j in 1:k if i != j)
    end

    k, T = 2, 800
    y, _, _ = generate_msm([0.0, 0.0], [0.05, 0.5], [[0.1], [0.2]], [[0.8], [0.6]], [0.97 0.03; 0.03 0.97], T)

    # SLSQP relies on the GARCH path's safeguards, so other models reject it
    @test_throws AssertionError MSModel(y, k, algorithm = :LD_SLSQP, verbose = false)
    @test_throws AssertionError MSARCHModel(y, k, 1, algorithm = :LD_SLSQP, verbose = false)

    # the previous default stays available for MS-GARCH
    m_sb = MSGARCHModel(y, k, 1, 1, intercept = "no", algorithm = :LN_SBPLX, maxtime = 10, verbose = false)
    @test m_sb isa MSM && isfinite(m_sb.Likelihood)

    # default (SLSQP) fit respects the raw-parameter caps, even from a user-supplied x0 beyond them
    m = MSGARCHModel(y, k, 1, 1, intercept = "no", verbose = false)
    @test all(m.raw_params[3:6] .<= MarSwitching.GARCH_RAW_CAP)
    @test all(m.raw_params[7:8] .<= MarSwitching.P_RAW_CAP)
    x0 = copy(m.raw_params); x0[3] = 10 * MarSwitching.GARCH_RAW_CAP; x0[7] = 10 * MarSwitching.P_RAW_CAP
    m_x0 = MSGARCHModel(y, k, 1, 1, intercept = "no", x0 = x0, maxtime = 10, verbose = false)
    @test isfinite(m_x0.Likelihood) && all(m_x0.raw_params[3:6] .<= MarSwitching.GARCH_RAW_CAP)
end

@testset "Less crucial functions" begin
    @test add_lags([1.0,2.0,3.0,4.0], 1) == [2.0 1.0; 3.0 2.0; 4.0 3.0]
    A = rand(4,4)
    @test all(abs.(MarSwitching.mp_inverse(A) .- pinv(A)) .< 0.001)

    # error density helpers
    y = [0.0, 1.0]
    μ = [0.0, 0.0]
    @test MarSwitching.error_density(:normal, y, μ, 1.0) isa Vector
    @test MarSwitching.error_density(:t, y, μ, 1.0, 5.0) isa Vector
    @test MarSwitching.error_density(:ged, y, μ, 1.0, 1.5) isa Vector
    @test_throws ArgumentError MarSwitching.error_density(:laplace, y, μ, 1.0)
end



