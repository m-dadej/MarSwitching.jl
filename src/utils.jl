
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

# 1:k                          - σ
# k+1:(k+k*n_β)                - state switching β aprameters
# (k+k*n_β+1):(k+k*n_β+n_β_ns) - non state switchign β parameters
# (k+k*n_β+n_β_ns+1):end        - transition probabilities


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
                 tvtp::Bool)
    
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
    
    return tvtp ? (σ, β) : (σ, β, P)
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


