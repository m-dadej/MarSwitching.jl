
"""
generate_mars(model::MSM, T::Int64)

When applied to estimated model, generates artificial data of size T from the model.
"""
function generate_mars(model::MSM, T::Int64)
    
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

"""
    generate_mars(μ::Vector{Float64}, σ::Vector{Float64}, P::Matrix{Float64}, T::Int64; <keyword arguments>)

Generate artificial data from Markov switching model from provided parameters.
Returns a tuple of `(y, s_t, X)` where `y` is the generated data, `s_t` is the state sequence and `X` is the design matrix.

Note, in order to have non-switching parameter provide it k-times.

# Arguments
- `μ::Vector{AbstractFloat}`: intercepts for each state.
- `σ::Vector{AbstractFloat}`: standard deviations for each state.
- `P::Matrix{AbstractFloat}`: transition matrix.
- `T::Int64`: number of observations to generate.
- `β::Vector{AbstractFloat}`: switching coefficients.
- `β_ns::Vector{AbstractFloat}`: non-switching coefficients.
- `δ::Vector{AbstractFloat}`: tvtp coefficients.
- `tvtp_intercept::Bool`: whether to include an intercept in the tvtp model.
"""
function generate_mars(μ::Vector{V},
                        σ::Vector{V},
                        P::Matrix{V},
                        T::Int64;
                        β::Vector{V} = Vector{V}([]),
                        β_ns::Vector{V} = Vector{V}([]),
                        δ::Vector{V} = Vector{V}([]),
                        tvtp_intercept::Bool = true) where V <: AbstractFloat
                        
    @assert size(μ)[1] == size(σ)[1] == size(P)[2] "size of μ, σ and P implies different number of states"
    @assert T > 0 "T should be a positive integer"

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
