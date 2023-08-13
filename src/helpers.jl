
function generate_mars(μ::Vector{Float64},
                       σ::Vector{Float64},
                       P::Matrix{Float64}, 
                       T::Int64,
                       seed::Int64)

    @assert size(P)[2] == length(μ) == length(σ) "Number of states not equal among provided parameters."

    if seed != 0 
        Random.seed!(seed)   
    end

    if size(P)[2] != size(P)[1]
        P = vcat(P, ones(1, size(P)[2]))
        P = P ./ sum(P, dims=1)
    end

    n_s = length(μ)
    s_t = [1]

    for _ in 1:(T-1)
        push!(s_t, sample(1:n_s, Weights(P[:, s_t[end]])))
    end

    x_s = zeros(T, n_s)

    for t in 1:T
        for s in 1:n_s
            x_s[t, s] = rand(Normal(μ[s], σ[s])) 
        end       
    end

    x = zeros(T)
    for s in 1:n_s
        x[s_t .== s] .= x_s[s_t .== s, s]
    end

    return x, s_t
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

function vec2param(θ::Vector{Float64}, k::Int64, n_β::Int64)
    
    σ = θ[1:k]
    β = [θ[k+1:(k+k*n_β)][1+n_β*i:n_β*(i+1)] for i in 0:k-1]

    P = reshape(θ[(k+k*n_β+1):end], k-1, k)

    return σ, β, P
end

function trans_θ(θ::Vector{Float64}, k::Int64, n_β::Int64)
    
    σ = θ[1:k].^2 
    β = [θ[k+1:(k+k*n_β)][1+n_β*i:n_β*(i+1)] for i in 0:k-1]

    @views P = reshape(θ[(k+k*n_β+1):end], k-1, k)
    P = [P; ones(1, k)]
    P = P ./ sum(P, dims=1)

    return σ, β, P
end






