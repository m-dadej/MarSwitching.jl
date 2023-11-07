
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

function Base.show(io::IO, ::MIME"text/plain", model::MSM)
    for s in 1:model.k
        print(io, "β_i,", s, ": ")
        [print(io, round(model.β[s][i], digits=3), " ") for i in 1:(model.n_β + model.n_β_ns + 1) ] 
        print(io, "\n------------------------------\n")
    end
    print(io, "σ = ", round.(model.σ, digits=3))
    println(io, "\n----------------------------")
    if !isempty(model.δ)
        print(io, "TVTP")
    else
        print(io, "P =")
        for i in 1:model.k
            for j in 1:model.k
                @printf io "%6s" round(model.P[i,j], digits=3) 
            end
            @printf io "%0s" "\n   "
        end 
    end        
    @printf io "%0s" "\nNLopt msg: $(model.nlopt_msg)"
end    

# Expectation-maximization algorithm for initial guess
function em_algorithm(X::VecOrMat, 
                      k::Int64,
                      n_β_ns::Int64,
                      n_δ::Int64,
                      n_intercept::Int64,
                      switching_var::Bool;
                      random_factor::Float64 = 0.5,
                      tol::Float64 = 1e-6)

    Q = [0.0, 1.0, 2.0, 3.0]
    y = X[:,1]
    x = X[:, 2:(end-n_δ)]
    x = n_intercept == 0 ? x[:, 2:end] : x
    T = size(y)[1]
    w = zeros(size(y)[1], k)

    # init_x = x \ y
    # β_hat = [init_x .+ (rand(Normal(0, 1)) .* random_factor) for _ in 1:k]

    β_hat = [rand(Normal(0, 1), size(x)[2]) for _ in 1:k]

    if n_intercept > 0
        [β_hat[i][1] = n_intercept == 1 ? 0.0 : rand(Normal(0, 1)) for i in 1:k]
    end  
    [β_hat[i][(end-n_β_ns+1):end] .= 0.0 for i in 1:k]

    σ_hat = [std(y) * (i/(k/2)) for i in 1:k]
    π_em = rand(k) 
    π_em = π_em ./ sum(π_em)
    
    while (Q[end] / Q[1] - 1) > tol
        ## Expectation step
        ϕ = hcat([pdf.(Normal.(x*β_hat[j], σ_hat[j]), y) for j in 1:k]...)
        ϕ .+= 1e-12
        w = (ϕ .* π_em') ./ sum(ϕ .* π_em', dims = 2)
        Q = my_circshift(Q, -1)
        Q[end] = sum(sum(w[i,j] .* ϕ[i,j] for j in 1:k) for i in 1:T)

        ## maximization step
        π_em  = (sum(w, dims=1) / T)'
        β_hat = [MarSwitching.mp_inverse(x'diagm(w[:,j])*x) * x'diagm(w[:,j])*y for j in 1:k]
        # averaging non-switching β
        β_ns_avrg = mean(reduce(hcat, β_hat)'[:, (end-n_β_ns+1):end], dims=1)
        [β_hat[i][(end-n_β_ns+1):end] = β_ns_avrg for i in 1:k]
        
        σ_hat = [sqrt(sum(w[:,j] .* (y .- x*β_hat[j]).^2) / sum(w[:,j])) for j in 1:k]
    end

    if n_intercept == 1
        intercept_avrg = mean(reduce(hcat, β_hat)'[:, 1])
        [β_hat[i][1] = intercept_avrg for i in 1:k]
    end

    σ_hat = switching_var ? σ_hat : (σ_hat'π_em)[:]

    return  π_em, β_hat, σ_hat, Q[end] 
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
            random_search::Int64 = 0,
            random_search_em::Int64,
            verbose::Bool) where V <: AbstractFloat   

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
- `random_search_em::Int64`: number of random searches to perform for the EM algorithm. If 0, no random search is performed.
- `random_search::Int64`: number of random searches to perform. 
- `verbose::Bool`: if true, prints out the progress of the random searches.

References:
- Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. Econometrica: Journal of the Econometric Society, 357-384.
- Filardo, Andrew J. (1994). Business cycle phases and their transitional dynamics. Journal of Business & Economic Statistics, 12(3), 299-308.

See also [`grid_search_msm`](@ref).
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
                 random_search_em::Int64 = 0,
                 random_search::Int64 = 0,
                 verbose::Bool = true) where V <: AbstractFloat              

    @assert size(y)[1] > 0 "y should be a vector or matrix with at least one observation"
    @assert k >= 2 "k should be at least 2, otherwise use standard linear regression"
    @assert intercept in ["switching", "non-switching", "no"] "intercept should be either 'switching', 'non-switching' or 'no'"
    @assert algorithm in [:LD_VAR2, :LD_VAR1, :LD_LBFGS, :LN_SBPLX] "algorithm should be either :LD_VAR2, :LD_VAR1, :LD_LBFGS, :LN_SBPLX"
    @assert (random_search_em >= 0) & (random_search >= 0) "Number of random searches for EM and optimization needs to be positive"

    # convert to matrix if vector
    exog_vars = typeof(exog_vars) <: Vector ? reshape(exog_vars, size(exog_vars)[1], 1) : exog_vars
    exog_switching_vars = typeof(exog_switching_vars) <: Vector ? reshape(exog_switching_vars, size(exog_switching_vars)[1], 1) : exog_switching_vars
    exog_tvtp = typeof(exog_tvtp) <: Vector ? reshape(exog_tvtp, size(exog_tvtp)[1], 1) : exog_tvtp

    T = size(y)[1]
    x = intercept == "no" ? [y zeros(T)] : [y ones(T)]

    ### counting number of variables ###
    n_β_ns = size(exog_vars)[2]                # number of non-switching β
    n_β    = size(exog_switching_vars)[2]      # number of switching β
    n_var  = switching_var ? k : 1             # number of variance parameters
    n_δ    = size(exog_tvtp)[2]                # number of tvtp terms in each state
    n_p    = n_δ > 0 ? n_δ*k*(k-1) : k*(k-1)   # number of probability parameters (either TVTP or constant)

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
    opt.xtol_rel      = 1e-4
    opt.maxtime       = maxtime < 0 ? T/2 : maxtime

    if n_δ == 0
        opt.min_objective = (θ, fΔ) -> obj_func(θ, fΔ, x, k, n_β, n_β_ns, intercept, switching_var)
    else
        opt.min_objective = (θ, fΔ) -> obj_func_tvtp(θ, fΔ, x, k, n_β, n_β_ns, intercept, switching_var, n_δ)
    end
    
    ### initial guess ###
    if isempty(x0)
        p_em_init, β_hat_init, σ_em_init, Q_init = em_algorithm(x, k, n_β_ns, n_δ, n_intercept, switching_var, random_factor = 0.0)

        ### random search for EM algorithm
        param_space = [[p_em_init, β_hat_init, σ_em_init, Q_init] for _ in 1:random_search_em+1]

        for i in 2:random_search_em+1
            param_space[i] .= em_algorithm(x, k, n_β_ns, n_δ, n_intercept, switching_var)
            verbose && println("EM algorithm random search: $(i-1) out of $random_search_em | Q = $(round.(param_space[i][end])) vs. Q_0 = $(round.(param_space[1][end]))")
        end

        [param_space[i][end] for i in 1:random_search_em+1]

        param_space = sort(param_space, by = x -> x[end], rev = false)
        (random_search_em > 0) & verbose && println("Q improvement with random search: $(round.(Q_init)) -> $(round.(param_space[end][end]))")
        p_em, β_hat, σ_em = param_space[end]

        ### transformation of ergodic probabilities to probabilites input to the optimization
        # this is bad code 
        # what i want to do is put the probabilites from EM algorithm into x0 anyhow
        if n_δ > 0
            p_em = ones(n_p)
            p_em[1:k:end] .= 1.5 # initial values with prior - diagonals are higher
        else
            pmat_em       = zeros(k,k)
            [pmat_em[i,i] = p_em[i] for i in 1:k]
            [pmat_em[i,j] = minimum(p_em) /2 for i in 1:k, j in 1:k if i != j]
            pmat_em       = pmat_em ./ sum(pmat_em, dims=1)
            pmat_em       = pmat_em[1:k-1, :] .* sum(pmat_em[1:k-1, :] .+ 1, dims=1) 
            p_em          = vec(pmat_em)    
        end

        ### converting initial values from EM to vector of parameters ###
        if intercept == "switching"
            μ_em = [β_hat[i][1] for i in 1:k]
        elseif intercept == "non-switching"
            μ_em = β_hat[1][1]
        elseif intercept == "no"
            μ_em = Vector{Float64}([])
        end

        β_ns_em = β_hat[1][(end-n_β_ns+1):end]
        β_s_em  = [β_hat[i][(end - n_β_ns - n_β+1):(end-n_β_ns)] for i in 1:k]
        β_s_em = vec(reduce(hcat, [β_s_em...]))

        x0 = [σ_em; μ_em; β_s_em; β_ns_em; p_em]
    end

    (minf_init, θ_hat_init, ret_init) = NLopt.optimize(opt, x0)

    ### Optimization random search ###
    param_space = [[minf_init, θ_hat_init, ret_init] for _ in 1:random_search+1]

    for i in 2:random_search+1
        rand_θ = param_space[1][2] .+ rand(Uniform(-0.5, 0.5), length(θ_hat_init))
        rand_θ = max.(opt.lower_bounds, rand_θ)

        param_space[i][1], param_space[i][2], param_space[i][3] = NLopt.optimize(opt, rand_θ)        
        verbose && println("Optimization random search: $(i-1) out of $random_search | LL = $(-round.(param_space[i][1]))")
    end

    param_space = sort(param_space, by = x -> x[1], rev = true)
    minf        = param_space[end][1]
    θ_hat       = param_space[end][2]
    ret         = param_space[end][3]
    (random_search > 0) & verbose && println("loglikelihood improvement with random search: $(-round.(minf_init)) -> $(-round.(param_space[end][1]))")

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

"""
    grid_search_msm(y::VecOrMat{V}, 
                    x::VecOrMat{V},
                    criterion::String = "AIC";
                    k::Vector{Int64} = [2,3,4],
                    intercept::Vector{String} = ["switching", "non-switching"],
                    vars::Vector{Vector{String}},
                    switching_var::Vector{Bool} = [true, false],
                    random_n::Int64,
                    random_search_em::Int64 = 0,
                    random_search::Int64 = 0,
                    verbose::Bool = true,
                    algorithm::Symbol = :LN_SBPLX,
                    maxtime::Int64 = -1) where V <: AbstractFloat  

Function for exhaustive or random search over specified parameter values for a Markov switching model (currently non-TVTP).
    
Returns a selected MSM model, vector of criterion values and a vector of tuples containing parameter space.

Note:
Unless the data is of small size (both dimensions), it is best to limit the parameter space by providing smaller possible parameters or by chosing random number of parameters to evaluate.

# Arguments
- `y::VecOrMat{V}`: dependent variable.
- `x::VecOrMat{V}`: independent variables.
- `criterion::String`: criterion to use for model selection. One of "AIC" (default) or "BIC".
- `k::Int64`: vector of states to evaluate.
- `intercept::String`: vector of "switching", "non-switching" or "no".
- `vars::Vector{Vector{String}}`: vector of vectors with either "switching" or "non-switching" for corresponding variables in `x` argument.
- `switching_var::Vector{Bool}`: vector of booleans for variance state dependency.
- `switching_var::Bool`: is variance state dependent?
- `random_n::Int64`: number of random parameters combinations to evaluate. If negative, performs an exhaustive grid search.
- `random_search_em::Int64`: number of random searches to perform for the EM algorithm in eery model estimation. If 0, no random search is performed.
- `random_search::Int64`: number of random searches to perform. 
- `algorithm::Symbol`: optimization algorithm to use. One of [:LD_VAR2, :LD_VAR1, :LD_LBFGS, :LN_SBPLX]
- `maxtime::Int64`: maximum time in seconds to run the optimization. If negative, the maximum time is equal T/2.
- `verbose::Bool`: if true, prints out the progress of the grid/random search.

See also [`MSModel`](@ref).
"""
function grid_search_msm(y::VecOrMat{V}, 
                        x::VecOrMat{V},
                        criterion::String = "AIC";
                        k::Vector{Int64} = [2,3,4],
                        intercept::Vector{String} = ["switching", "non-switching"],
                        vars::Vector{Vector{String}} = Vector{String}[],
                        switching_var::Vector{Bool} = [true, false],
                        algorithm::Symbol = :LN_SBPLX,
                        maxtime::Int64 = -1,
                        random_search_em::Int64 = 0,
                        random_search::Int64 = 0,
                        verbose::Bool = true,
                        random_n::Int64 = -1) where V <: AbstractFloat                          
                             
    x = typeof(x) <: Vector ? reshape(x, size(x)[1], 1) : x
    @assert size(y)[1] > 0 "y should be a vector or matrix with at least one observation"
    @assert all(k .>= 2) "k should be at least 2, otherwise use standard linear regression"
    @assert algorithm in [:LD_VAR2, :LD_VAR1, :LD_LBFGS, :LN_SBPLX] "algorithm should be either :LD_VAR2, :LD_VAR1, :LD_LBFGS, :LN_SBPLX"                   
    @assert all([intercept[i] in ["switching", "non-switching", "no"] for i in 1:length(intercept)]) "Possible parameters for intercept are [`switching`, `non-switching`, `no`]"
    @assert criterion in ["AIC", "BIC"] "Available criteria are `AIC` and `BIC`"
    @assert size(x)[1] == size(y)[1] "x and y should have the same number of observations"
    @assert (random_search_em >= 0) & (random_search >= 0) "Number of random searches for EM and optimization needs to be positive"

    vars = length(vars) == 0 ? [["switching", "non-switching"] for _ in 1:size(x)[2]] : vars                     
    @assert length(vars) == size(x)[2] "vars should be a vector of length equal to the number of columns in x"  
    @assert all([vars[i][j] in ["switching", "non-switching"] for i in 1:length(vars) for j in 1:length(vars[i])]) "Possible parameters for variables are `switching` or `non-switching`"

    vars_comb = vec(collect(Base.product(vars...)))
    n_combs = prod(size(Base.product(k, intercept, vars_comb, switching_var)))
    random_n = random_n == -1 ? n_combs : min(random_n, n_combs)
    param_space = vec(collect(Base.product(k, intercept, vars_comb, switching_var)))[sample(1:n_combs, random_n, replace = false)]

    models = Vector{MSM}(undef, size(param_space)[1])
    crit = Vector{Float64}(undef, size(param_space)[1])

    for i in 1:size(param_space)[1]

        models[i] = MSModel(y, 
                            param_space[i][1], 
                            intercept = param_space[i][2],
                            exog_vars = x[:, findall(param_space[i][3] .== "non-switching")],
                            exog_switching_vars = x[:, findall(param_space[i][3] .== "switching")],
                            switching_var = param_space[i][4],
                            algorithm = :LN_SBPLX,
                            maxtime = maxtime,
                            random_search_em = random_search_em,
                            random_search = random_search,
                            verbose = false)

        n_params  = length(models[i].raw_params)

        if criterion == "AIC"
            crit[i] = 2*n_params - 2*models[i].Likelihood
        elseif criterion == "BIC"
            crit[i] = log(models[i].T)*n_params - 2*models[i].Likelihood
        end                
        verbose && println("calculating combination $i/$(size(param_space)[1]) | criterion: $(round(crit[i], digits = 3))")                        
    end

    return models[findmin(crit)[2]], crit, param_space                   
end   
             
