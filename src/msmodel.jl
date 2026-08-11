
"""
Struct MSM holds the parameters of the model, data and some other information.
Is returned by the function `MSModel`.
"""
struct MSM{V <: AbstractFloat}
    β::Vector{Vector{V}}  # β[state][i] vector of β for each state
    σ::Vector{V}          # constant models: std dev; MS-ARCH: sqrt of mean conditional variance
    P::Matrix{V}          # transition matrix
    δ::Vector{V}          # tvtp parameters
    k::Int64
    n_β::Int64            # number of β parameters
    n_β_ns::Int64         # number of non-switching β parameters
    intercept::String     # "switching", "non-switching" or "no"
    switching_var::Bool   # is variance state dependent?
    q::Int64              # MS-ARCH order (0 = constant variance, i.e. a plain MSM)
    ω::Vector{V}          # MS-ARCH intercepts (variance units); empty when q == 0
    α::Vector{Vector{V}}  # α[state][lag], MS-ARCH coefficients; empty when q == 0
    error_dist::Symbol    # error distribution (:normal, :t, or :ged)
    ν::Vector{V}          # shape params for :t/:ged (empty for :normal)
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
    if model.q > 0
        print(io, "ω = ", round.(model.ω, digits=3))
        for j in 1:model.q
            print(io, "\nα_$j = ", round.([model.α[s][j] for s in 1:model.k], digits=3))
        end
        print(io, "\nσ (uncond.) = ", round.(model.σ, digits=3))
    else
        print(io, "σ = ", round.(model.σ, digits=3))
    end
    if has_shape_param(model.error_dist)
        print(io, "\nν = ", round.(model.ν, digits=3))
    end
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


# 1-D grid MLE for shape ν given residuals and observation weights
function mle_shape_1d(error_dist::Symbol, e::AbstractVector, σ::Float64, w::AbstractVector)
    σs = max(σ, 1e-8)
    sw = sum(w)
    sw <= 0 && return error_dist == :ged ? 2.0 : 5.0

    grid = error_dist == :t ? exp.(range(log(2.2), log(40.0), length=40)) :
                              exp.(range(log(0.7), log(5.0), length=40))

    best_ν, best_ll = grid[1], -Inf
    for ν in grid
        dens = error_density(error_dist, e, 0.0, σs, ν)
        ll = sum(@. w * log(max(dens, 1e-300)))
        if ll > best_ll
            best_ll = ll
            best_ν = ν
        end
    end
    return best_ν
end

# Moment-based starting value for ν from weighted excess kurtosis of residuals
function ν_from_kurtosis(error_dist::Symbol, e::AbstractVector, w::AbstractVector)
    sw = sum(w)
    sw <= 0 && return error_dist == :ged ? 2.0 : 5.0
    wn = w ./ sw
    m  = sum(wn .* e)
    e0 = e .- m
    m2 = sum(wn .* e0.^2)
    m4 = sum(wn .* e0.^4)
    m2 <= 1e-14 && return error_dist == :ged ? 2.0 : 5.0
    κ = m4 / m2^2 - 3.0   # excess kurtosis

    if error_dist == :t
        # excess kurtosis of t is 6/(ν-4) for ν>4
        if κ <= 0.05
            return 30.0
        else
            return clamp(4.0 + 6.0 / κ, 2.5, 40.0)
        end
    else
        # match GED excess kurtosis on a small grid
        best_ν, best_err = 2.0, Inf
        for ν in range(0.8, 4.5, length=40)
            # unit-GED kurtosis: Γ(5/ν)Γ(1/ν)/Γ(3/ν)^2 - 3
            g1 = exp(loggamma(1/ν))
            g3 = exp(loggamma(3/ν))
            g5 = exp(loggamma(5/ν))
            κ_g = g5 * g1 / g3^2 - 3.0
            err = abs(κ_g - κ)
            if err < best_err
                best_err = err
                best_ν = ν
            end
        end
        return best_ν
    end
end

# Expectation-maximization algorithm for initial guess
function em_algorithm(X::VecOrMat, 
                      k::Int64,
                      n_β_ns::Int64,
                      n_δ::Int64,
                      n_intercept::Int64,
                      switching_var::Bool,
                      error_dist::Symbol = :normal;
                      tol::Float64 = 1e-6,
                      maxiter::Int = 200)

    Q = [0.0, 1.0, 2.0, 3.0]
    y = X[:,1]
    x = X[:, 2:(end-n_δ)]
    x = n_intercept == 0 ? x[:, 2:end] : x
    T = size(y)[1]

    β_hat = [rand(Normal(0, 1), size(x)[2]) for _ in 1:k]

    if n_intercept > 0
        [β_hat[i][1] = n_intercept == 1 ? 0.0 : rand(Normal(0, 1)) for i in 1:k]
    end  
    [β_hat[i][(end-n_β_ns+1):end] .= 0.0 for i in 1:k]

    σ_hat = [std(y) * (i/(k/2)) for i in 1:k]
    ν_hat = has_shape_param(error_dist) ? fill(error_dist == :ged ? 2.0 : 6.0, k) : Float64[]
    π_em = rand(k) 
    π_em = π_em ./ sum(π_em)

    iter = 0
    while (Q[end] / Q[1] - 1) > tol && iter < maxiter
        iter += 1
        ## Expectation step — density matches the target error distribution
        ϕ = zeros(T, k)
        u = ones(T, k)   # latent scale weights (Student-t mixture); 1 for others
        for j in 1:k
            μj = x * β_hat[j]
            if error_dist == :normal
                ϕ[:, j] .= pdf.(Normal.(μj, σ_hat[j]), y)
            elseif error_dist == :t
                ej = y .- μj
                σj = max(σ_hat[j], 1e-8)
                νj = ν_hat[j]
                ϕ[:, j] .= error_density(:t, y, μj, σj, νj)
                # E[u | y, s=j] for t scale-mixture representation
                @. u[:, j] = (νj + 1.0) / (νj + (ej / σj)^2)
            else
                ϕ[:, j] .= error_density(:ged, y, μj, max(σ_hat[j], 1e-8), ν_hat[j])
            end
        end
        ϕ .+= 1e-12
        w = (ϕ .* π_em') ./ sum(ϕ .* π_em', dims = 2)
        Q = my_circshift(Q, -1)
        Q[end] = sum(dot(view(w, i, :), view(ϕ, i, :)) for i in 1:T)

        ## maximization step
        π_em  = (sum(w, dims=1) / T)'

        β_hat = Vector{Vector{Float64}}(undef, k)
        for j in 1:k
            # t: IRLS weights w * E[u]; normal/GED: state weights only
            wj = error_dist == :t ? (w[:, j] .* u[:, j]) : w[:, j]
            β_hat[j] = MarSwitching.mp_inverse((x'*(wj.*x))) * (x'*(wj.*y))
        end
        # averaging non-switching β
        β_ns_avrg = mean(reduce(hcat, β_hat)'[:, (end-n_β_ns+1):end], dims=1)
        [β_hat[i][(end-n_β_ns+1):end] = β_ns_avrg for i in 1:k]
        
        for j in 1:k
            ej = y .- x * β_hat[j]
            if error_dist == :t
                wj = w[:, j] .* u[:, j]
                swj = sum(w[:, j])
                σ_hat[j] = swj > 0 ? sqrt(max(sum(wj .* ej.^2) / swj, 1e-12)) : σ_hat[j]
            else
                swj = sum(w[:, j])
                σ_hat[j] = swj > 0 ? sqrt(max(sum(w[:, j] .* ej.^2) / swj, 1e-12)) : σ_hat[j]
            end
        end

        # shape parameters from residuals (moments warm-start + 1-D MLE polish)
        if has_shape_param(error_dist)
            for j in 1:k
                ej = y .- x * β_hat[j]
                ν_m = ν_from_kurtosis(error_dist, ej, w[:, j])
                ν_hat[j] = mle_shape_1d(error_dist, ej, σ_hat[j], w[:, j])
                # blend moment and MLE for stability early on
                ν_hat[j] = 0.5 * ν_hat[j] + 0.5 * ν_m
            end
        end
    end

    if n_intercept == 1
        intercept_avrg = mean(reduce(hcat, β_hat)'[:, 1])
        [β_hat[i][1] = intercept_avrg for i in 1:k]
    end

    σ_hat = switching_var ? σ_hat : (σ_hat'π_em)[:]
    if has_shape_param(error_dist) && !switching_var
        # single shared scale already; keep state-specific ν
    end

    return π_em, β_hat, σ_hat, ν_hat, Q[end] 
end


# Each of the four objective variants below calls its _const/_arch loglik body
# directly (bypassing the loglik()/loglik_tvtp() q>0 dispatch). MSModel() picks which
# variant to hand to NLopt ONCE, at setup time (see opt.min_objective below) — so the
# q=0 case, used on every optimizer iteration, has zero per-call dispatch overhead and
# matches the pre-MS-ARCH call depth exactly (obj_func_const -> _loglik_const, same as
# the original obj_func -> loglik).
function obj_func_const(θ, fΔ, x, k, n_β, n_β_ns, intercept, switching_var, error_dist)

    if length(fΔ) > 0
        fΔ[1:length(θ)] .= FiniteDiff.finite_difference_gradient(θ -> -_loglik_const(θ, x, k, n_β, n_β_ns, intercept, switching_var, error_dist, true)[1], θ)
    end

    return -_loglik_const(θ, x, k, n_β, n_β_ns, intercept, switching_var, error_dist, true)[1]
end

function obj_func_arch(θ, fΔ, x, k, n_β, n_β_ns, intercept, switching_var, error_dist, q)

    if length(fΔ) > 0
        fΔ[1:length(θ)] .= FiniteDiff.finite_difference_gradient(θ -> -_loglik_arch(θ, x, k, n_β, n_β_ns, intercept, switching_var, error_dist, q, true)[1], θ)
    end

    return -_loglik_arch(θ, x, k, n_β, n_β_ns, intercept, switching_var, error_dist, q, true)[1]
end

function obj_func_tvtp_const(θ, fΔ, x, k, n_β, n_β_ns, intercept, switching_var, n_δ, error_dist)

    if length(fΔ) > 0
        fΔ[1:length(θ)] .= FiniteDiff.finite_difference_gradient(θ -> -_loglik_tvtp_const(θ, x, k, n_β, n_β_ns, intercept, switching_var, n_δ, error_dist, true)[1], θ)
    end

    return -_loglik_tvtp_const(θ, x, k, n_β, n_β_ns, intercept, switching_var, n_δ, error_dist, true)[1]
end

function obj_func_tvtp_arch(θ, fΔ, x, k, n_β, n_β_ns, intercept, switching_var, n_δ, error_dist, q)

    if length(fΔ) > 0
        fΔ[1:length(θ)] .= FiniteDiff.finite_difference_gradient(θ -> -_loglik_tvtp_arch(θ, x, k, n_β, n_β_ns, intercept, switching_var, n_δ, error_dist, q, true)[1], θ)
    end

    return -_loglik_tvtp_arch(θ, x, k, n_β, n_β_ns, intercept, switching_var, n_δ, error_dist, q, true)[1]
end

"""
    MSModel(y::VecOrMat{V},
            k::Int64,
            ;intercept::String = "switching",
            exog_vars::VecOrMat{V},
            exog_switching_vars::VecOrMat{V},
            switching_var::Bool = true,
            q::Int64 = 0,
            error_dist::Symbol = :normal,
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
- `q::Int64`: order of the Markov-switching ARCH process (0 = constant variance, the default).
  See [`MSARCHModel`](@ref) for a convenience wrapper that sets this to a sensible non-zero default.
- `error_dist::Symbol`: distribution of the error term. One of `[:normal, :t, :ged]`.
  For `:t` / `:ged`, state-specific shape parameters ``\\nu`` are estimated
  (`:t` = degrees of freedom; `:ged` = GED shape with ``\\nu=2`` = normal, ``\\nu=1`` = Laplace).
- `exog_tvtp::VecOrMat{V}`: optional exogenous variables for the tvtp part of the model.

- `x0::Vector{V}`: initial guess for the parameters. If empty, the initial guess is generated from k-means clustering.
- `algorithm::Symbol`: optimization algorithm to use. One of [`:LD_VAR2`, `:LD_VAR1`, `:LD_LBFGS`, `:LN_SBPLX`]
- `maxtime::Int64`: maximum time in seconds to run the optimization. If negative, the maximum time is equal T/2.
- `random_search_em::Int64`: number of random searches to perform for the EM algorithm. If 0, no random search is performed.
- `random_search::Int64`: number of random searches to perform.
- `verbose::Bool`: if true, prints out the progress of the random searches.

References:
- Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. Econometrica: Journal of the Econometric Society, 357-384.
- Filardo, Andrew J. (1994). Business cycle phases and their transitional dynamics. Journal of Business & Economic Statistics, 12(3), 299-308.

See also [`grid_search_msm`](@ref), [`MSARCHModel`](@ref).
"""
function MSModel(y::VecOrMat{V},
                 k::Int64;
                 intercept::String = "switching", # or "non-switching"
                 exog_vars::VecOrMat{V} = Matrix{Float64}(undef, 0, 0),
                 exog_switching_vars::VecOrMat{V}= Matrix{Float64}(undef, 0, 0),
                 switching_var::Bool = true,
                 q::Int64 = 0,
                 error_dist::Symbol = :normal,
                 exog_tvtp::VecOrMat{V} = Matrix{Float64}(undef, 0, 0),
                 x0::Vector{V} = Vector{Float64}(undef, 0),
                 algorithm::Symbol = :LN_SBPLX,
                 maxtime::Int64 = -1,
                 random_search_em::Int64 = 0,
                 random_search::Int64 = 0,
                 verbose::Bool = true) where V <: AbstractFloat

    @assert size(y)[1] > 0 "y should be a vector or matrix with at least one observation"
    @assert k >= 2 "k should be at least 2, otherwise use standard linear regression"
    @assert q >= 0 "q should be non-negative"
    @assert intercept in ["switching", "non-switching", "no"] "intercept should be either 'switching', 'non-switching' or 'no'"
    @assert algorithm in [:LD_VAR2, :LD_VAR1, :LD_LBFGS, :LN_SBPLX] "algorithm should be either :LD_VAR2, :LD_VAR1, :LD_LBFGS, :LN_SBPLX"
    @assert error_dist in ERROR_DISTS "error_dist should be one of $ERROR_DISTS"
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
    n_ν    = has_shape_param(error_dist) ? k : 0  # shape params (:t df or :ged shape)
    n_α    = switching_var ? k*q : q           # MS-ARCH coefficients (0 when q == 0)
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
    n_params          = n_var + n_ν + n_α + n_β_ns + k*n_β + n_intercept + n_p
    @assert length(x0) == n_params || length(x0) == 0 "x0 should be either empty or of length $n_params"
    # also: LD_VAR2, :LD_VAR1, :LD_LBFGS, :LN_SBPLX
    opt               = Opt(algorithm, n_params)
    # σ/ω raw ≥ 0 (scale/variance = raw²); log(ν) unconstrained; α raw ≥ 0 (unbounded above); β unconstrained; P ≥ 0
    opt.lower_bounds  = [repeat([0.0], n_var); repeat([-Inf], n_ν); repeat([0.0], n_α);
                         repeat([-Inf], k*n_β + n_β_ns + n_intercept); repeat([n_δ > 0 ? -Inf : 0.0], n_p)]
    opt.xtol_rel      = (has_shape_param(error_dist) || q > 0) ? 1e-6 : 1e-4
    opt.maxtime       = maxtime < 0 ? T/2 : maxtime

    if n_δ == 0
        opt.min_objective = q > 0 ? (θ, fΔ) -> obj_func_arch(θ, fΔ, x, k, n_β, n_β_ns, intercept, switching_var, error_dist, q) :
                                    (θ, fΔ) -> obj_func_const(θ, fΔ, x, k, n_β, n_β_ns, intercept, switching_var, error_dist)
    else
        opt.min_objective = q > 0 ? (θ, fΔ) -> obj_func_tvtp_arch(θ, fΔ, x, k, n_β, n_β_ns, intercept, switching_var, n_δ, error_dist, q) :
                                    (θ, fΔ) -> obj_func_tvtp_const(θ, fΔ, x, k, n_β, n_β_ns, intercept, switching_var, n_δ, error_dist)
    end
    
    ### initial guess ###
    if isempty(x0)
        p_em_init, β_hat_init, σ_em_init, ν_em_init, Q_init =
            em_algorithm(x, k, n_β_ns, n_δ, n_intercept, switching_var, error_dist)

        ### random search for EM algorithm
        param_space = [[p_em_init, β_hat_init, σ_em_init, ν_em_init, Q_init] for _ in 1:random_search_em+1]

        for i in 2:random_search_em+1
            param_space[i] .= em_algorithm(x, k, n_β_ns, n_δ, n_intercept, switching_var, error_dist)
            verbose && println("EM algorithm random search: $(i-1) out of $random_search_em | Q = $(round.(param_space[i][end])) vs. Q_0 = $(round.(param_space[1][end]))")
        end

        param_space = sort(param_space, by = last, rev = false)
        (random_search_em > 0) & verbose && println("Q improvement with random search: $(round.(Q_init)) -> $(round.(last(param_space[end]))))")
        p_em, β_hat, σ_em, ν_em_hat = param_space[end][1:4]

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

        # raw ν = log(ν) from distribution-aware EM
        if has_shape_param(error_dist)
            ν_em = log.(max.(ν_em_hat, 1e-6))
        else
            ν_em = Float64[]
        end

        if q > 0
            # ω = raw² is a VARIANCE (unlike σ_raw² below, which is a std dev for the
            # constant-variance model): ω_s = (1-Σα)·σ_EM,s² gives an unconditional
            # variance ≈ σ_EM,s², matching the scale of the EM starting point.
            a0    = 0.3                                       # initial total ARCH persistence
            ω_raw = sqrt.((1 - a0) .* max.(σ_em, 1e-8).^2)
            α_raw = fill(sqrt(a0 / q), n_α)
            x0    = [ω_raw; ν_em; α_raw; μ_em; β_s_em; β_ns_em; p_em]
        else
            # raw σ: likelihood uses scale = raw², so pass sqrt(EM scale)
            σ_raw = sqrt.(max.(σ_em, 1e-8))
            x0    = [σ_raw; ν_em; μ_em; β_s_em; β_ns_em; p_em]
        end
    end

    (minf_init, θ_hat_init, ret_init) = NLopt.optimize(opt, x0)

    ### Optimization random search ###
    n_starts = random_search + 1
    # for shape distributions add structured multi-starts over ν (local maxima are common)
    ν_grid = if error_dist == :t
        [3.0, 5.0, 8.0, 12.0, 20.0]
    elseif error_dist == :ged
        [1.0, 1.3, 1.6, 2.0, 2.5, 3.5]
    else
        Float64[]
    end
    n_ν_starts = isempty(ν_grid) ? 0 : length(ν_grid)
    # structured multi-starts over total ARCH persistence Σα (MS-ARCH likelihoods are
    # classically bimodal: a high-variance regime can be explained by large ω or large α)
    α_grid      = q > 0 ? [0.05, 0.15, 0.35, 0.6] : Float64[]
    n_α_starts  = isempty(α_grid) ? 0 : length(α_grid)
    param_space = [[minf_init, θ_hat_init, ret_init] for _ in 1:(n_starts + n_ν_starts + n_α_starts)]

    for i in 2:n_starts
        # wider noise on log(ν) helps escape flat regions of the df parameters
        noise = rand(Uniform(-0.5, 0.5), length(θ_hat_init))
        if n_ν > 0
            noise[n_var+1:n_var+n_ν] .= rand(Uniform(-1.5, 1.5), n_ν)
        end
        if n_α > 0
            # narrower than the default noise: α_raw ≈ sqrt(0.3/q), so ±0.5 would sweep
            # Σα across an implausibly wide range
            noise[n_var+n_ν+1:n_var+n_ν+n_α] .= rand(Uniform(-0.2, 0.2), n_α)
        end
        rand_θ = param_space[1][2] .+ noise
        rand_θ = max.(opt.lower_bounds, rand_θ)

        param_space[i][1], param_space[i][2], param_space[i][3] = NLopt.optimize(opt, rand_θ)
        verbose && println("Optimization random search: $(i-1) out of $random_search | LL = $(-round.(param_space[i][1]))")
    end

    # structured ν multi-start: keep best β,σ,P and try distinct shape vectors
    if n_ν > 0 && n_ν_starts > 0
        base_θ = param_space[1][2]
        for (g, νg) in enumerate(ν_grid)
            θ_try = copy(base_θ)
            # spread shape across states around grid node
            ν_vec = [νg * (0.7 + 0.3*(j-1)/(max(k-1,1))) for j in 1:k]
            θ_try[n_var+1:n_var+n_ν] .= log.(ν_vec)
            idx = n_starts + g
            param_space[idx][1], param_space[idx][2], param_space[idx][3] = NLopt.optimize(opt, θ_try)
            verbose && println("Shape multi-start ν≈$νg | LL = $(-round.(param_space[idx][1]))")
        end
    end

    # structured α multi-start: keep best ω,β,P (and ν) and try distinct total-persistence levels
    if n_α > 0 && n_α_starts > 0
        base_θ = param_space[1][2]
        for (g, αg) in enumerate(α_grid)
            θ_try = copy(base_θ)
            θ_try[n_var+n_ν+1:n_var+n_ν+n_α] .= sqrt(αg / q)
            idx = n_starts + n_ν_starts + g
            param_space[idx][1], param_space[idx][2], param_space[idx][3] = NLopt.optimize(opt, θ_try)
            verbose && println("ARCH multi-start Σα≈$αg | LL = $(-round.(param_space[idx][1]))")
        end
    end

    param_space = sort(param_space, by = x -> x[1], rev = true)
    minf        = param_space[end][1]
    θ_hat       = param_space[end][2]
    ret         = param_space[end][3]
    (random_search > 0 || n_ν_starts > 0 || n_α_starts > 0) & verbose &&
        println("loglikelihood improvement with random search: $(-round.(minf_init)) -> $(-round.(param_space[end][1]))")

    ### transformation of variables - tvtp or not ###
    if n_δ > 0
        if q > 0
            ω, α, β, ν = trans_θ_arch(θ_hat, k, n_β, n_β_ns, intercept, switching_var, true, error_dist, q)
        else
            σ, β, ν = trans_θ(θ_hat, k, n_β, n_β_ns, intercept, switching_var, true, error_dist)
        end
        δ = θ_hat[(end-(n_δ*k*(k-1))+1):end]
        P = Matrix{Float64}(undef, 0, 0)
    else
        if q > 0
            ω, α, β, P, ν = trans_θ_arch(θ_hat, k, n_β, n_β_ns, intercept, switching_var, false, error_dist, q)
        else
            σ, β, P, ν = trans_θ(θ_hat, k, n_β, n_β_ns, intercept, switching_var, false, error_dist)
        end
        δ = Vector{Float64}(undef, 0)
    end

    if q > 0
        # sqrt(mean conditional variance): a single definition that is always finite and
        # nests the constant-variance model exactly (α = 0 ⇒ σ[s] = sqrt(ω_s))
        x_lik = n_δ > 0 ? x[:, 1:end-n_δ] : x   # mirrors the X-slicing in likelihood.jl
        μ_hat = [view(x_lik, :, 2:n_β+n_β_ns+2) * β[i] for i in 1:k]
        h_hat = arch_var(view(x_lik, :, 1), μ_hat, ω, α, k, q)
        σ     = [sqrt(sum(view(h_hat, :, s)) / T) for s in 1:k]
    else
        ω = Vector{Float64}(undef, 0)
        α = Vector{Vector{Float64}}(undef, 0)
    end

    return MSM(β, σ, P, δ, k, n_β, n_β_ns, intercept, switching_var, q, ω, α, error_dist, ν, x, T, -minf, θ_hat, ret)
end

"""
    MSARCHModel(y::VecOrMat{V},
               k::Int64,
               q::Int64 = 1;
               <same keyword arguments as MSModel>) where V <: AbstractFloat

Convenience wrapper around [`MSModel`](@ref) that estimates a Markov-Switching ARCH(q) model:
each of the `k` regimes carries its own ARCH(q) conditional-variance process
(Haas, Mittnik & Paolella, 2004), instead of a constant regime-specific variance.

```math
y_t = x_t'\\beta_{S_t} + e_t, \\qquad e_t = \\sqrt{h_{t,S_t}} \\, z_t, \\qquad z_t \\sim D(0,1)
```
```math
h_{t,s} = \\omega_s + \\sum_{j=1}^{q} \\alpha_{s,j} \\, \\varepsilon_{t-j,s}^2, \\qquad
\\varepsilon_{t,s} = y_t - x_t'\\beta_s
```

Each regime's conditional variance depends only on lags of that regime's own residual
`ε_{t,s}`, computed at every `t` for every state `s`, not just the realised one. This makes
the model path-independent (unlike Hamilton & Susmel's SWARCH, which requires expanding the
state space to `k^(q+1)`): `h` is a single `T×k` matrix computed once, and the Hamilton filter
is otherwise identical to [`MSModel`](@ref). It reduces exactly to [`MSModel`](@ref) when `α = 0`,
and with `intercept = "no"` and no exogenous variables it reduces to textbook Markov-switching
ARCH on `y` directly.

Note:
Pre-sample squared residuals (`t - j <= 0`) are backcast with the regime's full-sample mean
squared residual, rather than dropping the first `q` observations — this keeps `T`, and hence
AIC/BIC, comparable across different values of `q`.

For `error_dist = :t`, the Student-t is parameterised by scale, not standard deviation, so `h`
is a squared scale rather than a variance; the in-regime stationarity condition is
`Σⱼ α_{s,j}·ν_s/(ν_s-2) < 1`.

Standard errors for `α` degenerate towards the boundary `α = 0` (the score is identically zero
there), the same way `MSModel`'s finite-difference standard errors do at any other boundary.

# Arguments
- `y::VecOrMat{V}`: dependent variable.
- `k::Int64`: number of states.
- `q::Int64`: order of the ARCH process (same for every state). Must be at least 1.
- all other keyword arguments are identical to [`MSModel`](@ref), including `switching_var`
  (`true`: `ω` and `α` are state-specific; `false`: shared across states).

References:
- Haas, M., Mittnik, S., & Paolella, M. S. (2004). A new approach to Markov-switching GARCH models. Journal of Financial Econometrics, 2(4), 493-530.
- Hamilton, J. D., & Susmel, R. (1994). Autoregressive conditional heteroskedasticity and changes in regime. Journal of Econometrics, 64(1-2), 307-333.

See also [`MSModel`](@ref), [`conditional_variance`](@ref).
"""
function MSARCHModel(y::VecOrMat{V}, k::Int64, q::Int64 = 1; kwargs...) where V <: AbstractFloat
    @assert q >= 1 "q should be at least 1; use MSModel() for a constant-variance model"
    return MSModel(y, k; q = q, kwargs...)
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
                            algorithm = algorithm,
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
             
