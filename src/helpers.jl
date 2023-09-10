
function generate_mars(model::MSM, T::Int64 = model.T)
    
    μ    = [model.β[i][1] for i in 1:model.k]
    β    = [model.β[i][2:(2+model.n_β-1)] for i in 1:model.k]
    β    = vec(reduce(hcat, [β...]))
    β_ns = model.β[1][(2+model.n_β):end]
    
    δ = model.δ
    n_δ = Int(length(δ)/(model.k*(model.k-1)))
    exog_tvtp = model.x[:, end-n_δ+1:end]  
    tvtp_intercept = isempty(δ) ? false : all(exog_tvtp[:,1] .== exog_tvtp[1,1])

    return generate_mars(μ, model.σ, model.P, T, β = β, β_ns = β_ns, δ = δ, tvtp_intercept = tvtp_intercept)
end

function P_tvtp(x, δ, k, n_δ)

    P = reshape(exp.(reshape(δ, (k*(k-1)), n_δ)*x), k-1,k)
    P = [P; ones(1,k)]
    P = P ./ sum(P, dims=1)

    return P
end

function generate_mars(μ::Vector{Float64},
                        σ::Vector{Float64},
                        P::Matrix{Float64},
                        T::Int64;
                        β::Vector{Float64} = Vector{Float64}([]),
                        β_ns::Vector{Float64} = Vector{Float64}([]),
                        δ::Vector{Float64} = Vector{Float64}([]),
                        tvtp_intercept::Bool = true,
                        error_dist::String = "Normal",
                        v::Vector{Float64} = Vector{Float64}([]))

    if size(P)[2] != size(P)[1]
        P = vcat(P, ones(1, size(P)[2]))
        P = P ./ sum(P, dims=1)
    end

    k = length(μ)
    n_β = Int(size(β)[1]/k)
    n_β_ns = size(β_ns)[1]
    s_t = [1]

    if !isempty(δ)
        n_δ = Int(length(δ)/(k*(k-1)))
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

    params = [zeros(1 + n_β + n_β_ns) for _ in 1:k]
    [params[i][1] = μ[i] for i in 1:k]     # populate intercepts
    [params[i][2:(n_β+1)] .= β[1+n_β*(i-1):n_β*i] for i in 1:k] # populate switching betas
    [params[i][(n_β+2):(n_β+1+n_β_ns)] .= β_ns for i in 1:k] # populate switching betas

    for t in 1:T
        for s in 1:k
            if error_dist == "Normal"
                y_s[t, s] = rand(Normal((X*params[s])[t], σ[s]))
            elseif error_dist == "t"
                y_s[t, s] = (rand(TDist(v[s])) * σ[s]) + (X*params[s])[t]
            end                
            y_s[t, s] = rand(Normal((X*params[s])[t], σ[s])) 
        end       
    end

    X = !isempty(δ) ? [X x_tvtp] : X
    y = zeros(T)
    
    for s in 1:k
        y[s_t .== s] .= y_s[s_t .== s, s]
    end

    return y, s_t, X
end

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


# finite difference gradient and hessian, just in case

function finite_difference_gradient(f::Function, x::Vector{T}; h::T=1e-5) where T
    n = length(x)
    gradient = zeros(T, n)
    
    for i in 1:n
        x_forward = copy(x)
        x_backward = copy(x)
        
        x_forward[i] += h
        x_backward[i] -= h
        
        gradient[i] = (f(x_forward) - f(x_backward)) / (2 * h)
    end
    
    return gradient
end

function finite_difference_hessian(f::Function, x::Vector{T}; h::T=1e-5) where T
    n = length(x)
    hessian = zeros(T, n, n)
    
    gradient_x = finite_difference_gradient(f, x, h=h)
    
    for i in 1:n
        x_forward = copy(x)
        x_backward = copy(x)
        
        x_forward[i] += h
        x_backward[i] -= h
        
        gradient_forward = finite_difference_gradient(f, x_forward, h=h)
        gradient_backward = finite_difference_gradient(f, x_backward, h=h)
        
        hessian[:, i] = (gradient_forward - gradient_backward) / (2 * h)
    end
    
    return hessian
end

