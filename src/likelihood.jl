
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

function loglik(θ::Vector{Float64}, 
                X::Matrix{Float64}, 
                k::Int64,
                n_β::Int64,
                n_β_ns::Int64,
                intercept::String,
                switching_var::Bool,
                error_dist::Symbol=:normal,
                logsum::Bool=true)

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

function loglik_tvtp(θ::Vector{Float64}, 
                    X::Matrix{Float64}, 
                    k::Int64,
                    n_β::Int64,
                    n_β_ns::Int64,
                    intercept::String,
                    switching_var::Bool,
                    n_δ::Int64,
                    error_dist::Symbol=:normal,
                    logsum::Bool=true)

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
