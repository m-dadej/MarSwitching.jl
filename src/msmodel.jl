
"""
Struct MSM holds the parameters of the model, data and some other information.
Is returned by the function `MSModel`.
"""
struct MSM{V <: AbstractFloat}
    β::Vector{Vector{V}}  # β[state][i] vector of β for each state
    σ::Vector{V}         
    P::Matrix{V}          # transition matrix
    δ::Vector{V}          # tvtp parameters
    k::Int64                   
    n_β::Int64            # number of β parameters
    n_β_ns::Int64         # number of non-switching β parameters
    intercept::String     # "switching", "non-switching" or "no"
    switching_var::Bool   # is variance state dependent?
    x::Matrix{V}          # data matrix
    T::Int64              # number of observations
    Likelihood::Float64  
    raw_params::Vector{V} # raw parameters used directly in the Likelihood function
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

"""
    MSModel(y::VecOrMat{V},
            k::Int64, 
            ;intercept::String = "switching",
            exog_vars::VecOrMat{V},
            exog_switching_vars::VecOrMat{V},
            switching_var::Bool = true,
            exog_tvtp::VecOrMat{V},
            x0::Vector{V},
            algorithm::Symbol = :LN_SBPLX,
            maxtime::Int64 = -1,
            random_search::Int64 = 0) where V <: AbstractFloat   

Function to estimate the Markov Switching Model. Returns an instance of MSM struct.

Note:
The model likelihood function is very nonlinear and prone to local maxima. Increasing number of random searches can help, for the cost of longer training time.
For the same reason, it is recommended not to estimate model with many states (e.g. more than 5), altough it is possible.

# Arguments
- `y::VecOrMat{V}`: dependent variable.
- `k::Int64`: number of states.
- `intercept::String`: "switching" or "non-switching" or "no".
- `exog_vars::VecOrMat{V}`: optional exogenous variables for the non-switching part of the model.
- `exog_switching_vars::VecOrMat{V}`: optional exogenous variables for the switching part of the model.
- `switching_var::Bool`: is variance state dependent?
- `exog_tvtp::VecOrMat{V}`: optional exogenous variables for the tvtp part of the model.

- `x0::Vector{V}`: initial guess for the parameters. If empty, the initial guess is generated from k-means clustering.
- `algorithm::Symbol`: optimization algorithm to use. One of [:LD_VAR2, :LD_VAR1, :LD_LBFGS, :LN_SBPLX]
- `maxtime::Int64`: maximum time in seconds to run the optimization. If negative, the maximum time is equal T/2.
- `random_search::Int64`: number of random searches to perform. If 0, no random search is performed.

References:
- Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. Econometrica: Journal of the Econometric Society, 357-384.
- Filardo, Andrew J. (1994). Business cycle phases and their transitional dynamics. Journal of Business & Economic Statistics, 12(3), 299-308.

"""
function MSModel(y::VecOrMat{V},
                 k::Int64;
                 intercept::String = "switching", # or "non-switching"
                 exog_vars::VecOrMat{V} = Matrix{Float64}(undef, 0, 0),
                 exog_switching_vars::VecOrMat{V}= Matrix{Float64}(undef, 0, 0),
                 switching_var::Bool = true,
                 exog_tvtp::VecOrMat{V} = Matrix{Float64}(undef, 0, 0),
                 x0::Vector{V} = Vector{Float64}(undef, 0),
                 algorithm::Symbol = :LN_SBPLX,
                 maxtime::Int64 = -1,
                 random_search::Int64 = 0) where V <: AbstractFloat              

    @assert size(y)[1] > 0 "y should be a vector or matrix with at least one observation"
    @assert k >= 2 "k should be at least 2, otherwise use standard linear regression"
    @assert intercept in ["switching", "non-switching", "no"] "intercept should be either 'switching', 'non-switching' or 'no'"
    @assert algorithm in [:LD_VAR2, :LD_VAR1, :LD_LBFGS, :LN_SBPLX] "algorithm should be either :LD_VAR2, :LD_VAR1, :LD_LBFGS, :LN_SBPLX"

    # convert to matrix if vector
    exog_vars = typeof(exog_vars) <: Vector ? reshape(exog_vars, size(exog_vars)[1], 1) : exog_vars
    exog_switching_vars = typeof(exog_switching_vars) <: Vector ? reshape(exog_switching_vars, size(exog_switching_vars)[1], 1) : exog_switching_vars
    exog_tvtp = typeof(exog_tvtp) <: Vector ? reshape(exog_tvtp, size(exog_tvtp)[1], 1) : exog_tvtp

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
    n_params          = n_var + n_β_ns + k*n_β + n_intercept + n_p
    @assert length(x0) == n_params || length(x0) == 0 "x0 should be either empty or of length $n_params"
    # also: LD_VAR2, :LD_VAR1, :LD_LBFGS, :LN_SBPLX
    opt               = Opt(algorithm, n_params) 
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
