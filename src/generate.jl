
# Draw one innovation from the chosen error distribution
function sample_error(dist::Symbol, μ::Float64, σ::Float64, ν::Float64=Inf)
    if dist == :normal
        return rand(Normal(μ, σ))
    elseif dist == :t
        return μ + σ * rand(TDist(ν))
    elseif dist == :ged
        return rand_ged(μ, σ, ν)
    else
        throw(ArgumentError("Unsupported error distribution: $dist. Use :normal, :t, or :ged"))
    end
end

# X = μ + α S G^{1/ν}, G ~ Gamma(1/ν,1), S = ±1, α = σ √(Γ(1/ν)/Γ(3/ν))
function rand_ged(μ::Float64, σ::Float64, ν::Float64)
    νs = min(max(ν, 0.1), 50.0)
    α = σ * exp(0.5 * (loggamma(1 / νs) - loggamma(3 / νs)))
    g = rand(Gamma(1 / νs, 1.0))
    s = rand() < 0.5 ? -1.0 : 1.0
    return μ + α * s * g^(1 / νs)
end

"""
generate_msm(μ::Vector{Float64}, σ::Vector{Float64}, P::Matrix{Float64}, T::Int64; <keyword arguments>)

Generate artificial data from Markov switching model from provided parameters.
Returns a tuple of `(y, s_t, X)` where `y` is the generated data, `s_t` is the state sequence and `X` is the design matrix.

Note, in order to have non-switching parameter provide it k-times.

# Arguments
- `μ::Vector{AbstractFloat}`: intercepts for each state.
- `σ::Vector{AbstractFloat}`: standard deviations (scale) for each state.
- `P::Matrix{AbstractFloat}`: transition matrix.
- `T::Int64`: number of observations to generate.
- `β::Vector{AbstractFloat}`: switching coefficients. (first k elements in vector are coefficeints of the first state equation).
- `β_ns::Vector{AbstractFloat}`: non-switching coefficients.
- `δ::Vector{AbstractFloat}`: tvtp coefficients.
- `tvtp_intercept::Bool`: whether to include an intercept in the tvtp model.
- `error_dist::Symbol`: error distribution (`:normal`, `:t`, or `:ged`).
- `ν::Vector{AbstractFloat}`: shape parameters for each state when `error_dist` is `:t` (df) or `:ged` (shape; 2 = normal, 1 = Laplace).
"""
function generate_msm(μ::Vector{V},
                        σ::Vector{V},
                        P::Matrix{V},
                        T::Int64;
                        β::Vector{V} = Vector{V}([]),
                        β_ns::Vector{V} = Vector{V}([]),
                        δ::Vector{V} = Vector{V}([]),
                        tvtp_intercept::Bool = true,
                        error_dist::Symbol = :normal,
                        ν::Vector{V} = Vector{V}([])) where V <: AbstractFloat
                        
    isempty(δ) && @assert size(μ)[1] == size(σ)[1] == size(P)[2] "size of μ, σ and P implies different number of states"
    !isempty(δ) && @assert size(μ)[1] == size(σ)[1] "size of μ and σ implies different number of states"
    @assert T > 0 "T should be a positive integer"
    !isempty(β) && @assert length(β) % size(μ)[1] == 0 "β should be a multiple of k"
    @assert error_dist in ERROR_DISTS "error_dist should be one of $ERROR_DISTS"
    if has_shape_param(error_dist)
        @assert length(ν) == length(μ) "ν must have one element per state when error_dist = $error_dist"
        @assert all(ν .> 0) "shape parameters ν must be positive"
    end

    # for transition matrix without last row, add it and normalize
    if size(P)[2] != size(P)[1]
        P = vcat(P, ones(1, size(P)[2]))
        P = P ./ sum(P, dims=1)
    end

    k      = length(μ)
    n_β    = Int(size(β)[1]/k)
    n_β_ns = size(β_ns)[1]
    s_t    = [1]
    ν_s    = has_shape_param(error_dist) ? ν : fill(Inf, k)

    # generate random states either from tvtp or from P
    if !isempty(δ)

        n_δ = Int(length(δ)/(k*(k-1)))
        # generate tvtp exogenous variables from standard normal
        x_tvtp = tvtp_intercept ? [ones(T) rand(Normal(1,0.5), T, n_δ-1)] : rand(Normal(1,0.5), T, n_δ)

        for t in 1:(T-1)
            P = P_tvtp(x_tvtp[t, :], δ, k, n_δ)
            push!(s_t, sample(1:k, Weights(P[:, s_t[end]])))
        end
    else
        for _ in 1:(T-1)
            push!(s_t, sample(1:k, Weights(P[:, s_t[end]])))
        end
    end
    
    y_s = zeros(T, k)
    X = [ones(T) rand(Normal(0,1), T, n_β + n_β_ns)]

    # convert parameters to matrix for easier matrix multiplication
    params = [zeros(1 + n_β + n_β_ns) for _ in 1:k]
    [params[i][1] = μ[i] for i in 1:k]                          # populate intercepts
    [params[i][2:(n_β+1)] .= β[1+n_β*(i-1):n_β*i] for i in 1:k] # populate switching betas
    [params[i][(n_β+2):(n_β+1+n_β_ns)] .= β_ns for i in 1:k]    # populate switching betas

    # generate target variable for each state
    for t in 1:T
        for s in 1:k
            y_s[t, s] = sample_error(error_dist, (X*params[s])[t], σ[s], ν_s[s])
        end       
    end

    X = !isempty(δ) ? [X x_tvtp] : X
    y = zeros(T)
    
    # generate target variable from state specific target variables
    for s in 1:k
        y[s_t .== s] .= y_s[s_t .== s, s]
    end

    return y, s_t, X
end

"""
generate_msm(μ::Vector{Float64}, ω::Vector{Float64}, α::Vector{Vector{Float64}}, P::Matrix{Float64}, T::Int64; <keyword arguments>)

Generate artificial data from a Markov-Switching ARCH model (see [`MSARCHModel`](@ref)) from provided parameters.
Returns a tuple of `(y, s_t, X)` where `y` is the generated data, `s_t` is the state sequence and `X` is the design matrix.

Note, in order to have a non-switching parameter provide it k-times.

# Arguments
- `μ::Vector{AbstractFloat}`: intercepts for each state.
- `ω::Vector{AbstractFloat}`: ARCH intercepts (variance units) for each state.
- `α::Vector{Vector{AbstractFloat}}`: `α[state]` holds the ARCH coefficients (length `q`) for that state; every state must have the same `q`.
- `P::Matrix{AbstractFloat}`: transition matrix.
- `T::Int64`: number of observations to generate.
- `β::Vector{AbstractFloat}`: switching coefficients. (first k elements in vector are coefficeints of the first state equation).
- `β_ns::Vector{AbstractFloat}`: non-switching coefficients.
- `δ::Vector{AbstractFloat}`: tvtp coefficients.
- `tvtp_intercept::Bool`: whether to include an intercept in the tvtp model.
- `error_dist::Symbol`: error distribution (`:normal`, `:t`, or `:ged`).
- `ν::Vector{AbstractFloat}`: shape parameters for each state when `error_dist` is `:t` (df) or `:ged` (shape; 2 = normal, 1 = Laplace).
"""
function generate_msm(μ::Vector{V},
                      ω::Vector{V},
                      α::Vector{Vector{V}},
                      P::Matrix{V},
                      T::Int64;
                      β::Vector{V} = Vector{V}([]),
                      β_ns::Vector{V} = Vector{V}([]),
                      δ::Vector{V} = Vector{V}([]),
                      tvtp_intercept::Bool = true,
                      error_dist::Symbol = :normal,
                      ν::Vector{V} = Vector{V}([])) where V <: AbstractFloat

    isempty(δ) && @assert size(μ)[1] == size(ω)[1] == size(P)[2] "size of μ, ω and P implies different number of states"
    !isempty(δ) && @assert size(μ)[1] == size(ω)[1] "size of μ and ω implies different number of states"
    @assert T > 0 "T should be a positive integer"
    !isempty(β) && @assert length(β) % size(μ)[1] == 0 "β should be a multiple of k"
    @assert error_dist in ERROR_DISTS "error_dist should be one of $ERROR_DISTS"
    if has_shape_param(error_dist)
        @assert length(ν) == length(μ) "ν must have one element per state when error_dist = $error_dist"
        @assert all(ν .> 0) "shape parameters ν must be positive"
    end

    k = length(μ)
    @assert length(α) == k "α must hold one vector of ARCH coefficients per state"
    @assert all(length(α[s]) == length(α[1]) for s in 1:k) "every state needs the same ARCH order q"
    @assert all(ω .> 0) "ω must be positive"
    @assert all(all(α[s] .>= 0) for s in 1:k) "α must be non-negative"

    # for transition matrix without last row, add it and normalize
    if size(P)[2] != size(P)[1]
        P = vcat(P, ones(1, size(P)[2]))
        P = P ./ sum(P, dims=1)
    end

    q      = length(α[1])
    n_β    = Int(size(β)[1]/k)
    n_β_ns = size(β_ns)[1]
    s_t    = [1]
    ν_s    = has_shape_param(error_dist) ? ν : fill(Inf, k)

    # generate random states either from tvtp or from P
    if !isempty(δ)

        n_δ = Int(length(δ)/(k*(k-1)))
        # generate tvtp exogenous variables from standard normal
        x_tvtp = tvtp_intercept ? [ones(T) rand(Normal(1,0.5), T, n_δ-1)] : rand(Normal(1,0.5), T, n_δ)

        for t in 1:(T-1)
            P = P_tvtp(x_tvtp[t, :], δ, k, n_δ)
            push!(s_t, sample(1:k, Weights(P[:, s_t[end]])))
        end
    else
        for _ in 1:(T-1)
            push!(s_t, sample(1:k, Weights(P[:, s_t[end]])))
        end
    end

    X = [ones(T) rand(Normal(0,1), T, n_β + n_β_ns)]

    # convert parameters to matrix for easier matrix multiplication
    params = [zeros(1 + n_β + n_β_ns) for _ in 1:k]
    [params[i][1] = μ[i] for i in 1:k]                          # populate intercepts
    [params[i][2:(n_β+1)] .= β[1+n_β*(i-1):n_β*i] for i in 1:k] # populate switching betas
    [params[i][(n_β+2):(n_β+1+n_β_ns)] .= β_ns for i in 1:k]    # populate switching betas

    M = X * reduce(hcat, params)   # T×k conditional means, precomputed (avoids an O(T²k) loop)

    y  = zeros(T)
    e2 = zeros(T, k)               # e2[t,s] = (y_t - M[t,s])^2, ALL states, using the one realised y_t
    h  = zeros(T, k)
    # pre-sample variance: unconditional variance when the state is stationary, ω otherwise
    h0 = [sum(α[s]) < 1 ? ω[s] / (1 - sum(α[s])) : ω[s] for s in 1:k]

    # sequential: unlike the constant-variance method above, states cannot be drawn
    # independently per column, since every regime's variance path is driven by the
    # single realised y_t (see arch_var() in likelihood.jl for the estimation-side analogue)
    @inbounds for t in 1:T
        for s in 1:k
            v = ω[s]
            for j in 1:q
                v += α[s][j] * (t - j >= 1 ? e2[t-j, s] : h0[s])
            end
            h[t, s] = max(v, 1e-12)
        end
        s_now = s_t[t]
        y[t]  = sample_error(error_dist, M[t, s_now], sqrt(h[t, s_now]), ν_s[s_now])
        for s in 1:k
            e2[t, s] = (y[t] - M[t, s])^2
        end
    end

    X = !isempty(δ) ? [X x_tvtp] : X

    return y, s_t, X
end

"""
generate_msm(model::MSM, T::Int64)

When applied to estimated model, generates artificial data of size T from the model.
"""
function generate_msm(model::MSM, T::Int64 = 0)

    # extract parameters from model
    μ    = [model.β[i][1] for i in 1:model.k]
    β    = [model.β[i][2:(1+model.n_β)] for i in 1:model.k]
    β    = vec(reduce(hcat, [β...]))
    β_ns = model.β[1][(2+model.n_β):end]
    T    = T == 0 ? model.T : T

    # extract tvtp parameters from model
    δ = model.δ
    n_δ = Int(length(δ)/(model.k*(model.k-1)))
    exog_tvtp = model.x[:, end-n_δ+1:end]
    tvtp_intercept = isempty(δ) ? false : all(exog_tvtp[:,1] .== exog_tvtp[1,1])

    if model.q > 0
        return generate_msm(μ, model.ω, model.α, model.P, T, β = β, β_ns = β_ns, δ = δ,
                            tvtp_intercept = tvtp_intercept,
                            error_dist = model.error_dist, ν = model.ν)
    end

    return generate_msm(μ, model.σ, model.P, T, β = β, β_ns = β_ns, δ = δ,
                        tvtp_intercept = tvtp_intercept,
                        error_dist = model.error_dist, ν = model.ν)
end
