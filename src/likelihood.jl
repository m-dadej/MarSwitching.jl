
# Conditional density f(y | μ, σ, ν) for error_dist ∈ {:normal, :t, :ged}
function error_density(dist::Symbol, y, μ, σ::Float64, ν::Float64=Inf)
    σs = max(σ, 1e-10)  # numerical floor during optimization
    if dist == :normal
        return pdf.(Normal.(μ, σs), y)
    elseif dist == :t
        # location-scale Student-t: y = μ + σ * T(ν)
        νs = min(max(ν, 1e-6), 1e6)
        return pdf.(TDist(νs), (y .- μ) ./ σs) ./ σs
    elseif dist == :ged
        # unit-variance Generalized Error Distribution (exponential power):
        # f(z) ∝ exp(-|z/α|^ν) with α chosen so Var(z)=1, then y = μ + σ z
        # ν = 2 ⇒ Normal; ν = 1 ⇒ Laplace; ν < 2 heavier tails
        return ged_density(y, μ, σs, ν)
    else
        throw(ArgumentError("Unsupported error distribution: $dist. Use :normal, :t, or :ged"))
    end
end

# Unit-variance GED density (σ = std; ν = 2 ⇒ normal, ν = 1 ⇒ Laplace)
function ged_density(y, μ, σ::Float64, ν::Float64)
    νs = min(max(ν, 0.1), 50.0)
    # α = σ * sqrt(Γ(1/ν) / Γ(3/ν))  makes σ the standard deviation
    log_α = log(σ) + 0.5 * (loggamma(1 / νs) - loggamma(3 / νs))
    α = exp(log_α)
    # pdf = ν / (2 α Γ(1/ν)) * exp(-(|(y-μ)/α|)^ν)
    log_c = log(νs) - log(2.0) - log_α - loggamma(1 / νs)
    return @. exp(log_c - abs((y - μ) / α)^νs)
end

# Time-varying-scale overloads (MS-ARCH: σ is a per-observation vector, not a scalar).
# Kept as separate methods (rather than a single ::Union{Float64,AbstractVector} method)
# to match the existing style of the package, which prefers duplicated, dispatch-cheap
# methods over branching on the hot likelihood path (see comment in utils.jl).
function error_density(dist::Symbol, y, μ, σ::AbstractVector{Float64}, ν::Float64=Inf)
    σs = max.(σ, 1e-10)
    if dist == :normal
        return pdf.(Normal.(μ, σs), y)
    elseif dist == :t
        νs = min(max(ν, 1e-6), 1e6)
        return pdf.(TDist(νs), (y .- μ) ./ σs) ./ σs
    elseif dist == :ged
        return ged_density(y, μ, σs, ν)
    else
        throw(ArgumentError("Unsupported error distribution: $dist. Use :normal, :t, or :ged"))
    end
end

function ged_density(y, μ, σ::AbstractVector{Float64}, ν::Float64)
    νs    = min(max(ν, 0.1), 50.0)
    log_α = log.(σ) .+ 0.5 * (loggamma(1 / νs) - loggamma(3 / νs))
    α     = exp.(log_α)
    log_c = log(νs) - log(2.0) .- log_α .- loggamma(1 / νs)
    return @. exp(log_c - abs((y - μ) / α)^νs)
end

# T×k matrix of MS-ARCH(q) conditional variances (Haas, Mittnik & Paolella 2004).
# Each state s runs its own parallel ARCH(q) process off the state-s residual
# ε_{t,s} = y_t - μ_{t,s}, independent of the realised state path S_t. This
# path-independence is what lets h be computed in one vectorized pass, entirely
# outside the Hamilton filter recursion (exactly like the constant-σ case).
#
# Pre-sample ε²_{t-j} for t-j <= 0 is backcast with the regime's full-sample
# mean squared residual. h is floored AND ceiled: α is unbounded above during
# optimization, so an exploding path must not overflow to Inf (which would
# make the density silently zero rather than just very small).
function arch_var(y::AbstractVector{Float64},
                  μ::Vector{<:AbstractVector{Float64}},
                  ω::Vector{Float64},
                  α::Vector{Vector{Float64}},
                  k::Int64,
                  q::Int64)

    T  = length(y)
    e2 = Matrix{Float64}(undef, T, k)
    h  = Matrix{Float64}(undef, T, k)

    @inbounds for s in 1:k
        μs = μ[s]
        for t in 1:T
            e2[t, s] = (y[t] - μs[t])^2
        end
    end

    @inbounds for s in 1:k
        b = 0.0
        for t in 1:T
            b += e2[t, s]
        end
        b /= T                                  # backcast for t-j <= 0
        ωs, αs = ω[s], α[s]
        for t in 1:T
            v = ωs
            for j in 1:q
                v += αs[j] * (t - j >= 1 ? e2[t-j, s] : b)
            end
            h[t, s] = min(max(v, 1e-12), 1e12)
        end
    end

    return h
end

# Thin dispatcher only: q=0 and q>0 are handled by two fully self-contained functions
# below (a "function barrier"), rather than an if/else sharing β/σ/ω/P/ν bindings in
# one function body. trans_θ() and trans_θ_arch() each have their own internal runtime
# branch (tvtp) and so aren't fully type-inferred in general; merging their results into
# shared variables here would make Julia infer β/σ/ν as boxed (Core.Box) values, which
# cascades into μ and then η — the T×k density matrix read on every iteration of the
# filter loop below — becoming ::Any instead of ::Matrix{Float64}, at a large performance
# cost (measured ~6x on a 2-state/1-regressor model before this split). @inline is a
# no-op here in practice (Julia won't inline a call into a function containing a T-length
# loop regardless of the hint) but costs nothing to leave on. The hottest call site,
# MSModel()'s NLopt objective, doesn't go through this dispatcher at all — see
# obj_func_const()/obj_func_arch() in msmodel.jl, which call _loglik_const()/_loglik_arch()
# directly, since the q>0 choice is already known once at MSModel() setup time and
# re-checking it on every one of the (many) evaluations NLopt performs is pure overhead.
# This dispatcher exists for the less-hot call sites (get_std_errors, filtered_probs,
# tests) that want a single q-polymorphic entry point.
@inline function loglik(θ::Vector{Float64},
                X::Matrix{Float64},
                k::Int64,
                n_β::Int64,
                n_β_ns::Int64,
                intercept::String,
                switching_var::Bool,
                error_dist::Symbol=:normal,
                q::Int64=0,
                logsum::Bool=true)

    return q > 0 ? _loglik_arch(θ, X, k, n_β, n_β_ns, intercept, switching_var, error_dist, q, logsum) :
                   _loglik_const(θ, X, k, n_β, n_β_ns, intercept, switching_var, error_dist, logsum)
end

# constant-variance path (q == 0): identical to the pre-MS-ARCH implementation.
function _loglik_const(θ::Vector{Float64},
                       X::Matrix{Float64},
                       k::Int64,
                       n_β::Int64,
                       n_β_ns::Int64,
                       intercept::String,
                       switching_var::Bool,
                       error_dist::Symbol,
                       logsum::Bool)

    T      = size(X)[1]
    ξ      = zeros(T, k)  # unconditional transition probabilities at t
    L      = zeros(T)     # likelihood
    ξ_next = zeros(k)     # unconditional transition probabilities at t+1

    σ, β, P, ν = trans_θ(θ, k, n_β, n_β_ns, intercept, switching_var, false, error_dist)

    ξ_0 = ergodic_probs(P, k)
    ξ_0 = any(ξ_0 .< 0) ? ones(k) ./ k : ξ_0 # numerical stability check

    # f(y | S_t, x, θ, Ψ_t-1) density function
    y = view(X, :, 1)
    μ = [view(X, :, 2:n_β+n_β_ns+2) * β[i] for i in 1:k]
    η = reduce(hcat, [error_density(error_dist, y, μ[i], σ[i], isempty(ν) ? Inf : ν[i]) for i in 1:k])

    # if there is an underflow error for some reason I added this:
    #η .+= 1e-30
    # but it may alter the estimations in some cases. Same in TVTP

    @inbounds for t in 1:T
        ξ[t,:] = t == 1 ? ξ_0 : view(ξ, t-1, :)
        #ξ_next = P'ξ[t, :]
        #P = P_tvtp(x_tvtp[t], δ, k)
        mul!(ξ_next, P, view(ξ, t, :))  # same as: ξ_next  = P*view(ξ, t, :)
        L[t] = max(view(η, t, :)'ξ_next, 1e-300)
        @views @. ξ[t,:] = (1/L[t]) * ξ_next * η[t, :]
    end

    return (logsum ? sum(log.(L)) : L ), ξ #sum(log.(L)), ξ
end

# MS-ARCH path (q > 0): conditional variance is time-varying (arch_var()), so η is built
# from a per-observation scale vector instead of a per-state scalar. Otherwise identical
# in structure to _loglik_const above.
function _loglik_arch(θ::Vector{Float64},
                      X::Matrix{Float64},
                      k::Int64,
                      n_β::Int64,
                      n_β_ns::Int64,
                      intercept::String,
                      switching_var::Bool,
                      error_dist::Symbol,
                      q::Int64,
                      logsum::Bool)

    T      = size(X)[1]
    ξ      = zeros(T, k)
    L      = zeros(T)
    ξ_next = zeros(k)

    ω, α, β, P, ν = trans_θ_arch(θ, k, n_β, n_β_ns, intercept, switching_var, false, error_dist, q)

    ξ_0 = ergodic_probs(P, k)
    ξ_0 = any(ξ_0 .< 0) ? ones(k) ./ k : ξ_0 # numerical stability check

    y = view(X, :, 1)
    μ = [view(X, :, 2:n_β+n_β_ns+2) * β[i] for i in 1:k]
    h = arch_var(y, μ, ω, α, k, q)
    η = reduce(hcat, [error_density(error_dist, y, μ[i], sqrt.(view(h, :, i)), isempty(ν) ? Inf : ν[i]) for i in 1:k])

    @inbounds for t in 1:T
        ξ[t,:] = t == 1 ? ξ_0 : view(ξ, t-1, :)
        mul!(ξ_next, P, view(ξ, t, :))
        L[t] = max(view(η, t, :)'ξ_next, 1e-300)
        @views @. ξ[t,:] = (1/L[t]) * ξ_next * η[t, :]
    end

    return (logsum ? sum(log.(L)) : L ), ξ
end

# Thin dispatcher only — see the comment above loglik() for why this must not merge
# β/σ/ω/ν from trans_θ() and trans_θ_arch() into shared bindings in one function body.
@inline function loglik_tvtp(θ::Vector{Float64},
                    X::Matrix{Float64},
                    k::Int64,
                    n_β::Int64,
                    n_β_ns::Int64,
                    intercept::String,
                    switching_var::Bool,
                    n_δ::Int64,
                    error_dist::Symbol=:normal,
                    q::Int64=0,
                    logsum::Bool=true)

    return q > 0 ? _loglik_tvtp_arch(θ, X, k, n_β, n_β_ns, intercept, switching_var, n_δ, error_dist, q, logsum) :
                   _loglik_tvtp_const(θ, X, k, n_β, n_β_ns, intercept, switching_var, n_δ, error_dist, logsum)
end

# constant-variance path (q == 0): identical to the pre-MS-ARCH implementation.
function _loglik_tvtp_const(θ::Vector{Float64},
                            X::Matrix{Float64},
                            k::Int64,
                            n_β::Int64,
                            n_β_ns::Int64,
                            intercept::String,
                            switching_var::Bool,
                            n_δ::Int64,
                            error_dist::Symbol,
                            logsum::Bool)

    T      = size(X)[1]
    ξ      = zeros(T, k)  # unconditional transition probabilities at t
    L      = zeros(T)     # likelihood
    ξ_next = zeros(k)     # unconditional transition probabilities at t+1
    x_tvtp = X[:, end-n_δ+1:end]
    X      = X[:, 1:(end-n_δ)]

    #δ = θ[(end-(n_δ*k^2)+1):end]
    δ = θ[(end-(n_δ*k*(k-1))+1):end]
    σ, β, ν = trans_θ(θ, k, n_β, n_β_ns, intercept, switching_var, true, error_dist)

    # TO DO: use the same function as in the non-tvtp case but with tvtp
    ξ_0 = ones(k) ./ k

    # f(y | S_t, x, θ, Ψ_t-1) density function
    y = view(X, :, 1)
    μ = [view(X, :, 2:n_β+n_β_ns+2) * β[i] for i in 1:k]
    η = reduce(hcat, [error_density(error_dist, y, μ[i], σ[i], isempty(ν) ? Inf : ν[i]) for i in 1:k])
    #η .+= 1e-30

    @inbounds for t in 1:T
        ξ[t,:] = t == 1 ? ξ_0 : view(ξ, t-1, :)
        #ξ_next = P'ξ[t, :]
        P = P_tvtp(x_tvtp[t, :], δ, k, n_δ)
        mul!(ξ_next, P, view(ξ, t, :))  # same as: ξ_next  = P*view(ξ, t, :)
        L[t] = max(view(η, t, :)'ξ_next, 1e-300)
        @views @. ξ[t,:] = (1/L[t]) * ξ_next * η[t, :]
    end

    return (logsum ? sum(log.(L)) : L ), ξ #sum(log.(L)), ξ
end

# MS-ARCH path (q > 0). See _loglik_arch() for the non-TVTP analogue.
function _loglik_tvtp_arch(θ::Vector{Float64},
                           X::Matrix{Float64},
                           k::Int64,
                           n_β::Int64,
                           n_β_ns::Int64,
                           intercept::String,
                           switching_var::Bool,
                           n_δ::Int64,
                           error_dist::Symbol,
                           q::Int64,
                           logsum::Bool)

    T      = size(X)[1]
    ξ      = zeros(T, k)
    L      = zeros(T)
    ξ_next = zeros(k)
    x_tvtp = X[:, end-n_δ+1:end]
    X      = X[:, 1:(end-n_δ)]

    δ = θ[(end-(n_δ*k*(k-1))+1):end]
    ω, α, β, ν = trans_θ_arch(θ, k, n_β, n_β_ns, intercept, switching_var, true, error_dist, q)

    ξ_0 = ones(k) ./ k

    y = view(X, :, 1)
    μ = [view(X, :, 2:n_β+n_β_ns+2) * β[i] for i in 1:k]
    h = arch_var(y, μ, ω, α, k, q)
    η = reduce(hcat, [error_density(error_dist, y, μ[i], sqrt.(view(h, :, i)), isempty(ν) ? Inf : ν[i]) for i in 1:k])

    @inbounds for t in 1:T
        ξ[t,:] = t == 1 ? ξ_0 : view(ξ, t-1, :)
        P = P_tvtp(x_tvtp[t, :], δ, k, n_δ)
        mul!(ξ_next, P, view(ξ, t, :))
        L[t] = max(view(η, t, :)'ξ_next, 1e-300)
        @views @. ξ[t,:] = (1/L[t]) * ξ_next * η[t, :]
    end

    return (logsum ? sum(log.(L)) : L ), ξ
end
