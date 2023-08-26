
struct MSM 
    β::Vector{Vector{Float64}} # β[state][i] vector of β for each state
    σ::Vector{Float64}         
    P::Matrix{Float64}         # transition matrix
    k::Int64                   
    n_β::Int64                 # number of β parameters
    n_β_ns::Int64              # number of non-switching β parameters
    intercept::String          # "switching" or "non-switching"
    switching_var::Bool           # is variance state dependent?
    x::Matrix{Float64}         # data matrix
    T::Int64    
    Likelihood::Float64
    raw_params::Vector{Float64}
end


function loglik(θ::Vector{Float64}, 
                X::Matrix{Float64}, 
                k::Int64,
                n_β::Int64,
                n_β_ns::Int64,
                intercept::String,
                switching_var::Bool,
                logsum::Bool=true)

    T      = size(X)[1]
    ξ      = zeros(T, k)  # unconditional transition probabilities at t
    L      = zeros(T)     # likelihood 
    ξ_next = zeros(k)     # unconditional transition probabilities at t+1

    σ, β, P = trans_θ(θ, k, n_β, n_β_ns, intercept, switching_var)

    #initial guess for the unconditional probabilities
    A = [I - P; ones(k)']

    # check if A'A is invertible
    ξ_0 = !isapprox(det(A'A), 0) ? (inv(A'A)*A')[:,end] : ones(k) ./ k

    # f(y | S_t, x, θ, Ψ_t-1) density function 
    η = reduce(hcat, [pdf.(Normal.(view(X, :,2:n_β+n_β_ns+2)*β[i], σ[i]), view(X, :,1)) for i in 1:k])
    η .+= 1e-12

    @inbounds for t in 1:T
        ξ[t,:] = t == 1 ? ξ_0 : view(ξ, t-1, :)
        #ξ_next = P'ξ[t, :]
        mul!(ξ_next, P, view(ξ, t, :))  # same as: ξ_next  = P*view(ξ, t, :)
        L[t] = view(η, t, :)'ξ_next
        @views @. ξ[t,:] = (1/L[t]) * ξ_next * η[t, :]
    end

    return (logsum ? sum(log.(L)) : L ), ξ #sum(log.(L)), ξ
end

function obj_func(θ, fΔ, x, k, n_β, n_β_ns, intercept, switching_var)  
    
    if length(fΔ) > 0
        fΔ[1:length(θ)] .= FiniteDiff.finite_difference_gradient(θ -> -loglik(θ, x, k, n_β, n_β_ns, intercept, switching_var)[1], θ)
    end

    return -loglik(θ, x, k, n_β, n_β_ns, intercept, switching_var)[1]
end

function MSModel(y::Vector{Float64},
                 k::Int64, 
                 ;intercept::String = "switching", # or "non-switching"
                 exog_vars::Matrix{Float64} = Matrix{Float64}(undef, 0, 0),
                 exog_switching_vars::Matrix{Float64} = Matrix{Float64}(undef, 0, 0),
                 switching_var::Bool = true,
                 x0::Vector{Float64} = Vector{Float64}(undef, 0),
                 algorithm::Symbol = :LN_SBPLX,
                 maxtime::Int64 = -1)

    @assert k >= 0 "Amount of states shoould not be negative"

    T   = size(y)[1]
    x   = intercept == "no" ? [y zeros(T)] : [y ones(T)]

    # number of β parameters without intercept
    n_β_ns      = size(exog_vars)[2]                 # non-switching number of β
    n_β         = size(exog_switching_vars)[2]          # switching number of β
    n_var       = switching_var ? k : 1

    if intercept == "switching"
        n_intercept = k
    elseif intercept == "non-switching"
        n_intercept = 1
    elseif intercept == "no"
        n_intercept = 0
    end

    if !isempty(exog_switching_vars)
        @assert size(y)[1] == size(exog_switching_vars)[1] "Number of observations is not the same between y and exog_switching_vars"
        x = [x exog_switching_vars]
    end

    if !isempty(exog_vars)
        @assert size(y)[1] == size(exog_vars)[1] "Number of observations is not the same between y and exog_vars"
        x = [x exog_vars]
    end
    
    # also: LD_VAR2, :LD_VAR1, :LD_LBFGS, :LN_SBPLX
    
    opt               = Opt(algorithm, n_var + n_β_ns + k*n_β + n_intercept + (k-1)*k) 
    opt.lower_bounds  = [repeat([10e-10], n_var); repeat([-Inf], k*n_β + n_β_ns + n_intercept); repeat([10e-10], (k-1)*k)]
    opt.xtol_rel      = 0
    opt.maxtime       = maxtime < 0 ? T/2 : maxtime
    opt.min_objective = (θ, fΔ) -> obj_func(θ, fΔ, x, k, n_β, n_β_ns, intercept, switching_var)
    
    if isempty(x0)
        
        kmeans_res = kmeans(reshape(x[:,1], 1, T), k)

        if intercept == "switching"
            μ_em = kmeans_res.centers'[:]
            #μx0[1:n_β:n_β*k] .= μ_em[:] 
        elseif intercept == "non-switching"
            μ_em = mean(x[:,1])
            #μx0[1:n_β:n_β*k] .= μ_em[:]
        elseif intercept == "no"
            μ_em = Vector{Float64}([])
        end

        σ_em = switching_var ? [std(x[kmeans_res.assignments .== i, 1]) for i in 1:k] : std(x[:,1])
        p_em = [sum(kmeans_res.assignments .== i ) / T for i in 1:k]

        # this is really bad code 
        # what i want to do is put the probabilites from kmeans into x0 anyhow
        pmat_em       = zeros(k,k)
        [pmat_em[i,i] = p_em[i] for i in 1:k]
        [pmat_em[i,j] = minimum(p_em) /2 for i in 1:k, j in 1:k if i != j]
        pmat_em       = pmat_em ./ sum(pmat_em, dims=1)
        pmat_em       = pmat_em[1:k-1, :] .* sum(pmat_em[1:k-1, :] .+ 1, dims=1) 
        p_em          = vec(pmat_em)

        x0 = [σ_em; μ_em; zeros(n_β*k); zeros(n_β_ns); p_em]
        #x0 = [repeat([std(x[:,1])], k).^2; repeat([mean(x[:,1])], k*(size(x)[2]-1)); repeat([0.5],(k-1)*k)]
    end
    
    (minf,θ_hat,ret) = NLopt.optimize(opt, x0)
    
    println(ret)
    σ, β, P = trans_θ(θ_hat, k, n_β, n_β_ns, intercept, switching_var)
    
    return MSM(β, σ, P, k, n_β, n_β_ns, intercept, switching_var, x, T, -minf, θ_hat)
end

function filtered_probs(model::MSM;
                        y::Vector{Float64} = Vector{Float64}(undef, 0),
                        exog_vars::Matrix{Float64} = Matrix{Float64}(undef, 0, 0),
                        exog_switching_vars::Matrix{Float64} = Matrix{Float64}(undef, 0, 0)
                        )                       
    # TO DO:
    # - check if provided y and exogenous are used in the model
    # - check if y, exogenous have the same number of observations

    if isempty(exog_vars) & isempty(exog_switching_vars) & isempty(y)
        x = model.x
    else
        T = length(y)
        x = model.intercept == "no" ? [y zeros(T)] : [y ones(T)]        
        x = !isempty(exog_switching_vars) ? [x exog_switching_vars] : x
        x = !isempty(exog_vars) ? [x exog_vars] : x
    end

    ξ = loglik(model.raw_params, 
                x, 
                model.k, 
                model.n_β, 
                model.n_β_ns, 
                model.intercept,
                model.switching_var)[2]

    return ξ
end

function smoothed_probs(msm_model::MSM, 
                        y::Vector{Float64} = Vector{Float64}(undef, 0),
                        exog_vars::Matrix{Float64} = Matrix{Float64}(undef, 0, 0),
                        exog_switching_vars::Matrix{Float64} = Matrix{Float64}(undef, 0, 0),
    )
    
    if isempty(exog_vars) & isempty(exog_switching_vars) & isempty(y)
        ξ = filtered_probs(msm_model)
    else
        ξ = filtered_probs(msm_model, y, exog_vars, exog_switching_vars)   
    end
    
    T = msm_model.T
    P = msm_model.P

    ξ_T      = zeros(size(ξ))
    ξ_T[T,:] = ξ[T, :]

    for t in reverse(1:T-1)
        ξ_rate     = ξ_T[t+1, :] ./ (P*ξ[t, :])
        ξ_T[t, :] .= P' * ξ_rate .* ξ[t, :]
    end

    return ξ_T
end

function predict(model::MSM, 
                 insample::Bool = false;
                 y::Vector{Float64} = Vector{Float64}(undef, 0),
                 exog_vars::Matrix{Float64} = Matrix{Float64}(undef, 0, 0),
                 exog_switching_vars::Matrix{Float64} = Matrix{Float64}(undef, 0, 0))
    
    # TO DO:
    # - check if provided y and exogenous are used in the model
    # - check if y, exogenous have the same number of observations

    if isempty(exog_vars) & isempty(exog_switching_vars) & isempty(y)
        x = model.x[:,2:end]
        ξ_t = filtered_probs(model) 
    else
        T = length(y)
        x = model.intercept == "no" ? zeros(T) : ones(T)       
        x = !isempty(exog_switching_vars) ? [x exog_switching_vars] : x
        x = !isempty(exog_vars) ? [x exog_vars] : x

        ξ_t = filtered_probs(model, y = y, 
                                exog_vars = exog_vars, 
                                exog_switching_vars = exog_switching_vars) 
    end

    if insample
        ŷ_s = (x*hcat(model.β...))
    else
        ξ_t = (model.P * ξ_t')'[1:end-1,:]
        ŷ_s  = (x*hcat(model.β...))[2:end,:]
    end

    ŷ = sum(ŷ_s .* ξ_t, dims = 2)

    return ŷ, ξ_t
end



