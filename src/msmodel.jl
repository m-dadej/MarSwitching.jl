
struct MSM 
    β::Vector{Vector{Float64}} # β[state][i] vector of β for each state
    σ::Vector{Float64}         
    P::Matrix{Float64}         # transition matrix
    δ::Vector{Float64}         # tvtp parameters
    k::Int64                   
    n_β::Int64                 # number of β parameters
    n_β_ns::Int64              # number of non-switching β parameters
    intercept::String          # "switching" or "non-switching"
    switching_var::Bool           # is variance state dependent?
    x::Matrix{Float64}         # data matrix
    T::Int64    
    Likelihood::Float64
    raw_params::Vector{Float64}
    nlopt_msg::Symbol
end


function obj_func(θ, fΔ, x, k, n_β, n_β_ns, intercept, switching_var)  
    
    if length(fΔ) > 0
        fΔ[1:length(θ)] .= FiniteDiff.finite_difference_gradient(θ -> -loglik(θ, x, k, n_β, n_β_ns, intercept, switching_var)[1], θ)
    end

    return -loglik(θ, x, k, n_β, n_β_ns, intercept, switching_var)[1]
end

function obj_func_tvtp(θ, fΔ, x, k, n_β, n_β_ns, intercept, switching_var, n_δ)  
    
    if length(fΔ) > 0
        fΔ[1:length(θ)] .= FiniteDiff.finite_difference_gradient(θ -> -loglik_tvtp(θ, x, k, n_β, n_β_ns, intercept, switching_var, n_δ)[1], θ)
    end
    
    return -loglik_tvtp(θ, x, k, n_β, n_β_ns, intercept, switching_var, n_δ)[1]
end

function MSModel(y::Vector{Float64},
                 k::Int64, 
                 ;intercept::String = "switching", # or "non-switching"
                 exog_vars::Matrix{Float64} = Matrix{Float64}(undef, 0, 0),
                 exog_switching_vars::Matrix{Float64} = Matrix{Float64}(undef, 0, 0),
                 switching_var::Bool = true,
                 exog_tvtp::Matrix{Float64} = Matrix{Float64}(undef, 0, 0),
                 x0::Vector{Float64} = Vector{Float64}(undef, 0),
                 algorithm::Symbol = :LN_SBPLX,
                 maxtime::Int64 = -1,
                 random_search::Int64 = 0)

    @assert k >= 0 "Amount of states shoould not be negative"

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
    opt               = Opt(algorithm, n_var + n_β_ns + k*n_β + n_intercept + n_p) 
    opt.lower_bounds  = [repeat([0], n_var); repeat([-Inf], k*n_β + n_β_ns + n_intercept); repeat([n_δ > 0 ? -Inf : 0.0], n_p)]
    opt.xtol_rel      = 0
    opt.maxtime       = maxtime < 0 ? T/2 : maxtime

    if n_δ == 0
        opt.min_objective = (θ, fΔ) -> obj_func(θ, fΔ, x, k, n_β, n_β_ns, intercept, switching_var)
    else
        opt.min_objective = (θ, fΔ) -> obj_func_tvtp(θ, fΔ, x, k, n_β, n_β_ns, intercept, switching_var, n_δ)
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
        
        x0 = [σ_em; μ_em; zeros(n_β*k); zeros(n_β_ns); p_em]
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
        σ, β = trans_θ(θ_hat, k, n_β, n_β_ns, intercept, switching_var, true)
        δ = θ_hat[(end-(n_δ*k*(k-1))+1):end]
        P = Matrix{Float64}(undef, 0, 0)
    else
        σ, β, P = trans_θ(θ_hat, k, n_β, n_β_ns, intercept, switching_var, false)
        δ = Vector{Float64}(undef, 0)
    end
    
    return MSM(β, σ, P, δ, k, n_β, n_β_ns, intercept, switching_var, x, T, -minf, θ_hat, ret)
end
