
function P_tvtp(x, δ, k, n_δ)

    P = reshape(exp.(reshape(δ, (k*(k-1)), n_δ)*x), k-1,k)
    P = [P; ones(1,k)]
    P = P ./ sum(P, dims=1)

    return P
end

"""
    add_lags(y::Vector{Float64}, p::Int64)

Given a vector `y` of length `T`, returns a matrix of size `(T-p) x (p+1)` where the first column is `y[p+1:T]`, second column is `y[p:T-1]` and so on.    

"""
function add_lags(y::Vector{Float64}, p::Int64)

    @assert p >= 0 "p must be non-negative"

    T = size(y)[1]
    x = zeros(T-p, p+1)

    for i in 1:p+1
        x[:,i] .= y[p+2-i:T-i+1]
    end

    return x
end
##

# the parameters structure is as follows:
#
# 1:n_var                                      - σ  (n_var = k if switching_var, else 1)
#                                              - stored scale is raw² (positivity reparam.)
# n_var+1:n_var+n_ν                            - log(ν) shape for :t / :ged (n_ν = k if needed)
#                                              - transformed as ν = exp(raw)
# next                                         - intercept / switching β / non-switching β
# end                                          - transition probabilities (or TVTP δ)
#
# MS-ARCH(q) models (see trans_θ_arch below) insert one extra block *after* the
# ν block and before the intercept/β block:
#
# n_var+n_ν+1:n_var+n_ν+n_α                    - α raw (n_α = k*q if switching_var, else q)
#                                              - state-major; α = raw² (positivity reparam.)
#                                              - here "σ"/n_var above is instead "ω", a variance
#
# α is placed after ν (not before) so that every existing ν index expression
# (msmodel.jl random-search noise/multi-start, results.jl SE splice) keeps
# working unmodified when q == 0, and continues to address the correct ν
# entries even when q > 0.

# true if error distribution has a state-specific shape parameter ν
has_shape_param(error_dist::Symbol) = error_dist in (:t, :ged)

const ERROR_DISTS = (:normal, :t, :ged)

# Lanczos approximation to logΓ(z) for z > 0 (avoids a SpecialFunctions dependency).
# Used only for GED normalizing constants — O(1) per state, not per observation.
const _LG_G = 5.0
const _LG_C = (
    1.000000000190015,
    76.18009172947146,
    -86.50532032941677,
    24.01409824083091,
    -1.231739572450155,
    0.1208650973866179e-2,
    -0.5395239384953e-5,
)

function loggamma(z::Real)
    z = Float64(z)
    z <= 0 && throw(DomainError(z, "loggamma requires positive argument"))
    x = z
    y = x
    tmp = x + _LG_G + 0.5
    tmp = (x + 0.5) * log(tmp) - tmp
    ser = _LG_C[1]
    @inbounds for i in 2:7
        y += 1.0
        ser += _LG_C[i] / y
    end
    return tmp + log(2.5066282746310005 * ser / x)
end


function vec2param_switch(θ::Vector{Float64}, 
                          k::Int64, 
                          n_β::Int64, 
                          n_β_ns::Int64,
                          switching_var::Bool)
    
    n_var = switching_var ? k : 1
    σ     = switching_var ? θ[1:k] : repeat([θ[1]], k)

    # make Vector{Vector{Float64}} of β each containing n_β + n_β_ns elements
    β = [zeros(n_β + n_β_ns + 1) for _ in 1:k]
    
    # fill the first n_β elements with state-switching parameters + intercept (either also if ns)
    [β[i][1] = θ[n_var+1:(n_var+k)][i] for i in 1:k]
    [β[i+1][2:n_β+1] .= θ[n_var+k+1:(n_var+k + n_β*k)][1+(n_β)*i:(n_β)*(i+1)] for i in 0:k-1]
    
    # fill the rest of the vectors with non-switching parameters (same for each state)
    if n_β_ns > 0
        [β[i][end-n_β_ns+1:end] .= θ[(n_var + k + n_β*k)+1:(n_var + k + n_β*k) + n_β_ns] for i in 1:k]
    end

    return σ, β
end

function vec2param_nonswitch(θ::Vector{Float64}, 
                             k::Int64, 
                             n_β::Int64, 
                             n_β_ns::Int64,
                             switching_var::Bool)

    n_var = switching_var ? k : 1
    σ     = switching_var ? θ[1:k] : repeat([θ[1]], k)

    # make Vector{Vector{Float64}} of β each containing n_β + n_β_ns elements
    β = [zeros(n_β + n_β_ns + 1) for _ in 1:k]
    
    [β[i][1] = θ[n_var+1:(n_var+1+k*n_β)][1] for i in 1:k]
    [β[i+1][2:n_β+1] .= θ[n_var+2:(n_var+1+k*n_β)][1+(n_β)*i:(n_β)*(i+1)] for i in 0:k-1]
    
    # fill the rest of the vectors with non-switching parameters (same for each state)
    if n_β_ns > 0
        [β[i][end-n_β_ns+1:end] .= θ[(n_var+1+n_β*k+1):(n_var+1+n_β*k+n_β_ns)] for i in 1:k]
    end

    return σ, β
end

# the same function as above, but without [β[i][1] = θ[k+1:(k+1+k*n_β)][1] for i in 1:k] and indexes moved
function vec2param_nointercept(θ::Vector{Float64}, 
                               k::Int64, 
                               n_β::Int64, 
                               n_β_ns::Int64,
                               switching_var::Bool)
    
    n_var = switching_var ? k : 1
    σ     = switching_var ? θ[1:k] : repeat([θ[1]], k)

    # make Vector{Vector{Float64}} of β each containing n_β + n_β_ns elements
    β = [zeros(n_β + n_β_ns + 1) for _ in 1:k]
    
    [β[i+1][2:n_β+1] .= θ[n_var+1:(n_var+k*n_β)][1+(n_β)*i:(n_β)*(i+1)] for i in 0:k-1]
    
    # fill the rest of the vectors with non-switching parameters (same for each state)
    if n_β_ns > 0
        [β[i][end-n_β_ns+1:end] .= θ[(n_var+n_β*k+1):(n_var+n_β*k+n_β_ns)] for i in 1:k]
    end

    return σ, β
end

function trans_θ(θ::Vector{Float64},
                 k::Int64,
                 n_β::Int64, 
                 n_β_ns::Int64, 
                 intercept::String,
                 switching_var::Bool,
                 tvtp::Bool,
                 error_dist::Symbol = :normal)
    
    n_var = switching_var ? k : 1
    n_ν   = has_shape_param(error_dist) ? k : 0

    # drop ν parameters so existing vec2param_* helpers keep working
    # raw parameter is unconstrained log-shape: ν = exp(θ) > 0
    if n_ν > 0
        ν_raw = θ[n_var+1:n_var+n_ν]
        ν     = exp.(ν_raw)
        θ     = vcat(θ[1:n_var], θ[n_var+n_ν+1:end])
    else
        ν = Vector{Float64}(undef, 0)
    end

    # I know, it should be done in a single function. But it's faster apparently.
    if intercept == "switching"
        σ, β = vec2param_switch(θ, k, n_β, n_β_ns, switching_var)
    elseif intercept == "non-switching"
        σ, β = vec2param_nonswitch(θ, k, n_β, n_β_ns, switching_var)
    elseif intercept == "no"
        σ, β = vec2param_nointercept(θ, k, n_β, n_β_ns, switching_var)
    end

    σ = σ.^2

    if !tvtp
        @views P = reshape(θ[end-(k*(k-1) - 1):end], k-1, k)
        P = [P; ones(1, k)]
        P = P ./ sum(P, dims=1)
    end
    
    return tvtp ? (σ, β, ν) : (σ, β, P, ν)
end

# MS-ARCH(q) parameter vector transform (Haas, Mittnik & Paolella 2004).
# Splices the α block out of θ, leaving a vector in exactly the layout that
# trans_θ() already knows how to parse, and delegates to it for everything
# else. This keeps trans_θ() and the vec2param_* helpers completely untouched.
#
# Returns (ω, α, β, ν) when tvtp, else (ω, α, β, P, ν).
# ω is a variance (raw², like σ² in trans_θ). α[state] is a length-q vector of
# ARCH coefficients (raw², state-major in the raw vector, unbounded above).
function trans_θ_arch(θ::Vector{Float64},
                      k::Int64,
                      n_β::Int64,
                      n_β_ns::Int64,
                      intercept::String,
                      switching_var::Bool,
                      tvtp::Bool,
                      error_dist::Symbol,
                      q::Int64)

    n_var = switching_var ? k : 1
    n_ν   = has_shape_param(error_dist) ? k : 0
    n_α   = switching_var ? k*q : q
    off   = n_var + n_ν

    α_raw = θ[off+1:off+n_α]
    θ_c   = vcat(θ[1:off], θ[off+n_α+1:end])

    α = Vector{Vector{Float64}}(undef, k)
    if switching_var
        [α[s] = α_raw[(s-1)*q+1:s*q].^2 for s in 1:k]
    else
        a = α_raw.^2
        [α[s] = copy(a) for s in 1:k]   # copy: each element stored independently on the struct
    end

    if tvtp
        ω, β, ν = trans_θ(θ_c, k, n_β, n_β_ns, intercept, switching_var, true, error_dist)
        return ω, α, β, ν
    else
        ω, β, P, ν = trans_θ(θ_c, k, n_β, n_β_ns, intercept, switching_var, false, error_dist)
        return ω, α, β, P, ν
    end
end

# function to shift the vector - circshift() equivalent
# circshift does not work for stable julia 1.6
function my_circshift(x::Vector{Float64}, n::Int64)
    if n > 0
        return [x[end-n+1:end]; x[1:end-n]]
    else
        return [x[-n+1:end]; x[1:-n]]
    end
end

# function to calculate moores-penrose pseudoinverse
# Function pinv() can't be used because the package won't be compatible with Julia 1.6
# anyway it's slightly but significantly faster than pinv() in benchmarks
function mp_inverse(A)
    U, S, V = svd(A)
    Σ = zeros(size(A'))
    Σ[1:size(S)[1], 1:size(S)[1]] = Diagonal(1 ./ S)
    return V * Σ * U'
end   

# function below combines vec2param_nonswitch and vec2param_switch
# apparently, it's slower than the two separate functions. Even though it's more concise.

# function vec2param(θ::Vector{Float64}, k::Int64, n_β::Int64, n_β_ns::Int64, intercept::String)
    
#     σ = θ[1:k]
#     # make Vector{Vector{Float64}} of β each containing n_β + n_β_ns elements
#     β = [zeros(n_β + n_β_ns + 1) for _ in 1:k]
    
#     if intercept == "non-switching"
#         last_β = (k+k*n_β + 1)
#         [β[i][1] = θ[k+1:(k+k*n_β + 1)][1] for i in 1:k]
#         [β[i+1][2:n_β+1] .= θ[k+1:last_β][(2 + n_β*i):(1 + n_β*(i+1))] for i in 0:k-1]
#     else
#         last_β = (k*2 + n_β*k)
#         # fill the first n_β elements with state-switching parameters + intercept (either also if ns)
#         [β[i+1][1:n_β+1] .= θ[k+1:last_β][1+(n_β+1)*i:(n_β+1)*(i+1)] for i in 0:k-1]
#     end
    
#     # fill the rest of the vectors with non-switching parameters (same for each state)
#     [β[i][end-n_β_ns+1:end] .= θ[last_β+1:last_β + n_β_ns] for i in 1:k]

#     @views P = reshape(θ[end-(k*(k-1) - 1):end], k-1, k)

#     return σ, β, P
# end


