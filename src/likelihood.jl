
struct MSM 
    β::Vector{Vector{Float64}} # β[state][i] vector of β for each state
    σ::Vector{Float64}         
    P::Matrix{Float64}         # transition matrix
    δ::Vector{Float64}         # tvtp parameters
    v::Vector{Float64}
    k::Int64                   
    n_β::Int64                 # number of β parameters
    n_β_ns::Int64              # number of non-switching β parameters
    intercept::String          # "switching" or "non-switching"
    switching_var::Bool        # is variance state dependent?
    error_dist::String
    x::Matrix{Float64}         # data matrix
    T::Int64    
    Likelihood::Float64
    raw_params::Vector{Float64}
    nlopt_msg::Symbol
end

function loglik(θ::Vector{Float64}, 
                    X::Matrix{Float64}, 
                    k::Int64,
                    n_β::Int64,
                    n_β_ns::Int64,
                    intercept::String,
                    switching_var::Bool,
                    error_dist::String,
                    logsum::Bool=true)

    T      = size(X)[1]
    ξ      = zeros(T, k)  # unconditional transition probabilities at t
    L      = zeros(T)     # likelihood 
    ξ_next = zeros(k)     # unconditional transition probabilities at t+1

    v = error_dist == "Normal" ? Vector{Float64}([]) : θ[end-k+1:end]
    σ, β, P = trans_θ(θ, k, n_β, n_β_ns, intercept, switching_var, false)
    
    # if tvtp
    #     P = trans_tvtp(P) 
    # end

    #initial guess for the unconditional probabilities
    A = [I - P; ones(k)']

    # check if A'A is invertible
    ξ_0 = !isapprox(det(A'A), 0) ? (inv(A'A)*A')[:,end] : ones(k) ./ k
    # numerical stability check
    ξ_0 = any(ξ_0 .< 0) ? ones(k) ./ k : ξ_0

    # f(y | S_t, x, θ, Ψ_t-1) density function 
    if error_dist == "Normal"
        η = reduce(hcat, [pdf.(Normal.(view(X, :,2:n_β+n_β_ns+2)*β[i], σ[i]), view(X, :,1)) for i in 1:k])
    elseif error_dist == "t"
        η = reduce(hcat, [pdf.((TDist(abs(v[i]))) , (view(X, :,1) .- view(X, :,2:n_β+n_β_ns+2)*β[i]) ./ σ[i]) for i in 1:k])
    elseif error_dist == "GEV"
        η = reduce(hcat, [pdf.(GeneralizedExtremeValue.(view(X, :,2:n_β+n_β_ns+2)*β[i], σ[i], v[i]), view(X, :,1)) for i in 1:k])
    elseif errr_dist == "GED"
        η = reduce(hcat, [pdf.(GeneralizedErrorDistribution.(view(X, :,2:n_β+n_β_ns+2)*β[i], σ[i], v[i]), view(X, :,1)) for i in 1:k])
    end        

    η .+= 1e-12

    @inbounds for t in 1:T
        ξ[t,:] = t == 1 ? ξ_0 : view(ξ, t-1, :)
        #ξ_next = P'ξ[t, :]
        #P = P_tvtp(x_tvtp[t], δ, k)
        mul!(ξ_next, P, view(ξ, t, :))  # same as: ξ_next  = P*view(ξ, t, :)
        L[t] = view(η, t, :)'ξ_next
        @views @. ξ[t,:] = (1/L[t]) * ξ_next * η[t, :]
    end

    return (logsum ? sum(log.(L)) : L ), ξ #sum(log.(L)), ξ
end

function loglik_tvtp(θ::Vector{Float64}, 
                    X::Matrix{Float64}, 
                    k::Int64,
                    n_β::Int64,
                    n_β_ns::Int64,
                    intercept::String,
                    switching_var::Bool,
                    n_δ::Int64,
                    error_dist::String,
                    logsum::Bool=true)

    T      = size(X)[1]
    ξ      = zeros(T, k)  # unconditional transition probabilities at t
    L      = zeros(T)     # likelihood 
    ξ_next = zeros(k)     # unconditional transition probabilities at t+1
    x_tvtp = X[:, end-n_δ+1:end]
    X      = X[:, 1:(end-n_δ)]
    
    #δ = θ[(end-(n_δ*k^2)+1):end]
    v = error_dist == "Normal" ? Vector{Float64}([]) : θ[end-k+1:end]
    δ = θ[(end-(n_δ*k*(k-1))+1 - length(v)):(end - length(v))]
    σ, β = trans_θ(θ, k, n_β, n_β_ns, intercept, switching_var, true)

    # TO DO: use the same function as in the non-tvtp case but with tvtp
    ξ_0 = ones(k) ./ k
    
    # η = f(y | S_t, x, θ, Ψ_t-1) density function 
    if error_dist == "Normal"
        η = reduce(hcat, [pdf.(Normal.(view(X, :,2:n_β+n_β_ns+2)*β[i], σ[i]), view(X, :,1)) for i in 1:k])
    elseif error_dist == "t"
        η = reduce(hcat, [pdf.((TDist(abs(v[i]))) , (view(X, :,1) .- view(X, :,2:n_β+n_β_ns+2)*β[i]) ./ σ[i]) for i in 1:k])
    elseif error_dist == "GEV"
        η = reduce(hcat, [pdf.(GeneralizedExtremeValue.(view(X, :,2:n_β+n_β_ns+2)*β[i], σ[i], v[i]), view(X, :,1)) for i in 1:k])
    elseif errr_dist == "GED"
        η = reduce(hcat, [pdf.(GeneralizedErrorDistribution.(view(X, :,2:n_β+n_β_ns+2)*β[i], σ[i], v[i]), view(X, :,1)) for i in 1:k])
    end
    η .+= 1e-12

    @inbounds for t in 1:T
        ξ[t,:] = t == 1 ? ξ_0 : view(ξ, t-1, :)
        #ξ_next = P'ξ[t, :]
        P = P_tvtp(x_tvtp[t, :], δ, k, n_δ)
        mul!(ξ_next, P, view(ξ, t, :))  # same as: ξ_next  = P*view(ξ, t, :)
        L[t] = view(η, t, :)'ξ_next
        @views @. ξ[t,:] = (1/L[t]) * ξ_next * η[t, :]
    end

    return (logsum ? sum(log.(L)) : L ), ξ #sum(log.(L)), ξ
end

# TO DO: add kwargs
function obj_func(θ, fΔ, x, k, n_β, n_β_ns, intercept, switching_var, error_dist)  
    
    if length(fΔ) > 0
        fΔ[1:length(θ)] .= FiniteDiff.finite_difference_gradient(θ -> -loglik(θ, x, k, n_β, n_β_ns, intercept, switching_var, error_dist)[1], θ)
    end

    return -loglik(θ, x, k, n_β, n_β_ns, intercept, switching_var, error_dist)[1]
end

function obj_func_tvtp(θ, fΔ, x, k, n_β, n_β_ns, intercept, switching_var, n_δ, error_dist)  
    
    if length(fΔ) > 0
        fΔ[1:length(θ)] .= FiniteDiff.finite_difference_gradient(θ -> -loglik_tvtp(θ, x, k, n_β, n_β_ns, intercept, switching_var, n_δ, error_dist)[1], θ)
    end
    
    return -loglik_tvtp(θ, x, k, n_β, n_β_ns, intercept, switching_var, n_δ, error_dist)[1]
end

function MSModel(y::Vector{Float64},
                 k::Int64, 
                 ;intercept::String = "switching", # or "non-switching"
                 exog_vars::Matrix{Float64} = Matrix{Float64}(undef, 0, 0),
                 exog_switching_vars::Matrix{Float64} = Matrix{Float64}(undef, 0, 0),
                 switching_var::Bool = true, 
                 error_dist::String = "Normal",
                 exog_tvtp::Matrix{Float64} = Matrix{Float64}(undef, 0, 0),
                 x0::Vector{Float64} = Vector{Float64}(undef, 0),
                 algorithm::Symbol = :LN_SBPLX,
                 maxtime::Int64 = -1,
                 random_search::Int64 = 0)

    @assert k >= 0 "Amount of states should not be negative"

    T   = size(y)[1]
    x   = intercept == "no" ? [y zeros(T)] : [y ones(T)]

    ### counting number of variables ###
    n_β_ns      = size(exog_vars)[2]                # number of non-switching β
    n_β         = size(exog_switching_vars)[2]      # number of switching β
    n_var       = switching_var ? k : 1             # number of variance parameters
    n_δ         = size(exog_tvtp)[2]                # number of tvtp terms in each state
    n_p         = n_δ > 0 ? n_δ*k*(k-1) : k*(k-1)   # number of probability parameters (either TVTP or constant)

    # number of intercept parameters
    if intercept == "switching"
        n_intercept = k
    elseif intercept == "non-switching"
        n_intercept = 1
    elseif intercept == "no"
        n_intercept = 0
    end

    n_dist_p = error_dist == "Normal" ? 0 : k

    ### merging dataset ###
    if !isempty(exog_switching_vars)
        @assert size(y)[1] == size(exog_switching_vars)[1] "Number of observations is not the same between y and exog_switching_vars"
        x = [x exog_switching_vars]
    end

    if !isempty(exog_vars)
        @assert size(y)[1] == size(exog_vars)[1] "Number of observations is not the same between y and exog_vars"
        x = [x exog_vars]
    end

    if !isempty(exog_tvtp)
        @assert size(y)[1] == size(exog_tvtp)[1] "Number of observations is not the same between y and exog_switching_vars"
        x = [x exog_tvtp]
    end
    
    ### solver settings ###
    # also: LD_VAR2, :LD_VAR1, :LD_LBFGS, :LN_SBPLX
    opt               = Opt(algorithm, n_var + n_β_ns + k*n_β + n_intercept + n_p + n_dist_p) 
    opt.lower_bounds  = [repeat([0], n_var); repeat([-Inf], k*n_β + n_β_ns + n_intercept); repeat([n_δ > 0 ? -Inf : 0.0], n_p); repeat([error_dist == "t" ? 1e-5 : -Inf], n_dist_p)]
    opt.xtol_rel      = 0
    opt.maxtime       = maxtime < 0 ? T/2 : maxtime

    if n_δ == 0
        opt.min_objective = (θ, fΔ) -> obj_func(θ, fΔ, x, k, n_β, n_β_ns, intercept, switching_var, error_dist)
    else
        opt.min_objective = (θ, fΔ) -> obj_func_tvtp(θ, fΔ, x, k, n_β, n_β_ns, intercept, switching_var, n_δ, error_dist)
    end
    
    ### initial guess ###
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
        if n_δ > 0
            p_em = ones(n_p)
        else
            pmat_em       = zeros(k,k)
            [pmat_em[i,i] = p_em[i] for i in 1:k]
            [pmat_em[i,j] = minimum(p_em) /2 for i in 1:k, j in 1:k if i != j]
            pmat_em       = pmat_em ./ sum(pmat_em, dims=1)
            pmat_em       = pmat_em[1:k-1, :] .* sum(pmat_em[1:k-1, :] .+ 1, dims=1) 
            p_em          = vec(pmat_em)    
        end
        
        x0 = [σ_em; μ_em; zeros(n_β*k); zeros(n_β_ns); p_em; (ones(n_dist_p) .* 1e-4) ]
        #x0 = [repeat([std(x[:,1])], k).^2; repeat([mean(x[:,1])], k*(size(x)[2]-1)); repeat([0.5],(k-1)*k)]
    end

    (minf_init, θ_hat_init, ret_init) = NLopt.optimize(opt, x0)

    ### random search ###
    param_space = [[minf_init, θ_hat_init, ret_init] for _ in 1:random_search+1]

    for i in 2:random_search+1
        rand_θ = param_space[1][2] .+ rand(Uniform(-0.5, 0.5), length(θ_hat_init))
        rand_θ = max.(opt.lower_bounds, rand_θ)

        param_space[i][1], param_space[i][2], param_space[i][3] = NLopt.optimize(opt, rand_θ)        
        println("random search: $(i-1) out of $random_search")
    end

    param_space = sort(param_space, by = x -> x[1], rev = true)
    random_search > 0 && println("loglikelihood improvement with random search: $(-round.(minf_init)) -> $(-round.(param_space[end][1]))")
    minf        = param_space[end][1]
    θ_hat       = param_space[end][2]
    ret         = param_space[end][3]

    ### transformation of variables - tvtp or not ###
    if n_δ > 0
        v = error_dist == "t" ? θ_hat[end-k+1:end] : Vector{Float64}([])
        δ = θ_hat[(end-(n_δ*k*(k-1))+1 - length(v)):(end - length(v))]

        σ, β = trans_θ(θ_hat, k, n_β, n_β_ns, intercept, switching_var, true)
        P = Matrix{Float64}(undef, 0, 0)
    else
        v = error_dist == "t" ? θ_hat[end-k+1:end] : Vector{Float64}([])
        σ, β, P = trans_θ(θ_hat, k, n_β, n_β_ns, intercept, switching_var, false)
        δ = Vector{Float64}(undef, 0)
    end
    
    return MSM(β, σ, P, δ, v, k, n_β, n_β_ns, intercept, switching_var, error_dist, x, T, -minf, θ_hat, ret)
end

function filtered_probs(model::MSM;
                        y::Vector{Float64} = Vector{Float64}(undef, 0),
                        exog_vars::Matrix{Float64} = Matrix{Float64}(undef, 0, 0),
                        exog_switching_vars::Matrix{Float64} = Matrix{Float64}(undef, 0, 0),
                        exog_tvtp::Matrix{Float64} = Matrix{Float64}(undef, 0, 0))                       
    # TO DO:
    # - check if provided y and exogenous are used in the model
    # - check if y, exogenous have the same number of observations

    if isempty(exog_vars) & isempty(exog_switching_vars) & isempty(y) & isempty(exog_tvtp)
        x = model.x
    else
        T = length(y)
        x = model.intercept == "no" ? [y zeros(T)] : [y ones(T)]        
        x = !isempty(exog_switching_vars) ? [x exog_switching_vars] : x
        x = !isempty(exog_vars) ? [x exog_vars] : x
        x = !isempty(exog_tvtp) ? [x exog_tvtp] : x
    end

    if !isempty(model.P)
        ξ = loglik(model.raw_params, 
                    x, 
                    model.k, 
                    model.n_β, 
                    model.n_β_ns, 
                    model.intercept,
                    model.switching_var)[2]
    else
        ξ = loglik_tvtp(model.raw_params, 
                        x, 
                        model.k, 
                        model.n_β,
                        model.n_β_ns, 
                        model.intercept, 
                        model.switching_var, 
                        Int(length(model.δ)/(model.k*(model.k-1))))[2]
    end
    

    return ξ
end


function smoothed_probs(model::MSM; kwargs...)
    
    
    P   = model.P
    k   = model.k
    δ   = model.δ
    n_δ = Int(length(δ)/(k*(k-1)))
                        
    if isempty(kwargs)
        ξ = filtered_probs(model)
        exog_tvtp = isempty(P) ? model.x[:, end-n_δ+1:end] : nothing
        T = model.T
    else
        ξ = filtered_probs(model; kwargs...)   
        exog_tvtp = isempty(P) ? kwargs[:exog_tvtp] : nothing 
        T = size(ξ)[1]
    end
    
    ξ_T      = zeros(size(ξ))
    ξ_T[T,:] = ξ[T, :]

    for t in reverse(1:T-1)
        P          = isempty(P) ? P_tvtp(exog_tvtp[t, :],δ, k, n_δ) : P
        ξ_rate     = ξ_T[t+1, :] ./ (P*ξ[t, :])
        ξ_T[t, :] .= P' * ξ_rate .* ξ[t, :]
    end

    return ξ_T
end

function predict(model::MSM, 
                 instanteous::Bool = false;
                 y::Vector{Float64} = Vector{Float64}(undef, 0),
                 exog_vars::Matrix{Float64} = Matrix{Float64}(undef, 0, 0),
                 exog_switching_vars::Matrix{Float64} = Matrix{Float64}(undef, 0, 0),
                 exog_tvtp::Matrix{Float64} = Matrix{Float64}(undef, 0, 0))
    
    # TO DO:
    # - check if provided y and exogenous are used in the model
    # - check if y, exogenous have the same number of observations

    if isempty(exog_vars) & isempty(exog_switching_vars) & isempty(y) & isempty(exog_tvtp)
        ξ_t = filtered_probs(model) 
        T = model.T
        n_δ = Int(length(model.δ)/(model.k*(model.k-1)))
        exog_tvtp = model.x[:, end-n_δ+1:end]
        x = model.x[:,2:end-n_δ]
    else
        T = length(y)
        x = model.intercept == "no" ? zeros(T) : ones(T)       
        x = !isempty(exog_switching_vars) ? [x exog_switching_vars] : x
        x = !isempty(exog_vars) ? [x exog_vars] : x

        ξ_t = filtered_probs(model, y = y, 
                                    exog_vars = exog_vars, 
                                    exog_switching_vars = exog_switching_vars,
                                    exog_tvtp = exog_tvtp) 
    end

    if instanteous
        ŷ_s = (x*hcat(model.β...))
    else
        if isempty(model.P)
            for t in 1:T
                P = P_tvtp(exog_tvtp[t, :], model.δ, model.k, Int(length(model.δ)/(model.k*(model.k-1))))
                ξ_t[t, :] = P'ξ_t[t, :]
            end
            ξ_t = ξ_t[1:end-1,:]
        else
            ξ_t = (model.P * ξ_t')'[1:end-1,:]
        end
        ŷ_s  = (x*hcat(model.β...))[2:end,:]
    end

    ŷ = sum(ŷ_s .* ξ_t, dims = 2)

    return ŷ, ξ_t
end



