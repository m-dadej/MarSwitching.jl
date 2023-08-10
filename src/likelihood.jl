using Distributions
using LinearAlgebra
using NLopt
using Random
using StatsBase
using FiniteDiff
using LineSearches
using Clustering
using Printf

struct MSM 
    β::Vector{Vector{Float64}} # β[state][i] vector of β for each state
    σ::Vector{Float64}         
    P::Matrix{Float64}         # transition matrix
    rawP::Vector{Float64}      # raw probabilites vector before [P 1] / sum(P) transformation
    k::Int64                   
    p::Int64                   # number of lags
    x::Matrix{Float64}         # data matrix
    T::Int64    
    Likelihood::Float64
end

include("helpers.jl")
include("results.jl")


function trans_θ(θ::Vector{Float64}, k::Int64, n_β::Int64)
    
    σ = θ[1:k].^2 
    β = [θ[k+1:(k+k*n_β)][1+n_β*i:n_β*(i+1)] for i in 0:k-1]

    @views P = reshape(θ[(k+k*n_β+1):end], k-1, k)
    P        = [P; ones(1, k)]
    P        = P ./ sum(P, dims=1)

    return σ, β, P
end


function loglik(θ::Vector{Float64}, 
                X::Matrix{Float64}, 
                k::Int64,
                logsum::Bool=true)

    T      = size(X)[1]
    n_β    = size(X)[2]-1 # number of β coefficients. Including lagged y
    ξ      = zeros(T, k)  # unconditional transition probabilities at t
    L      = zeros(T)     # likelihood 
    ξ_next = zeros(k)  

    σ, β, P = trans_θ(θ, k, n_β)

    #initial guess for the unconditional probabilities
    A = [I - P; ones(k)']

    # check if A'A is invertible
    ξ_0 = !isapprox(det(A'A), 0) ? (inv(A'A)*A')[:,end] : ones(k) ./ k

    # f(y | S_t, x, θ, Ψ_t-1) density function 
    η = reduce(hcat, [pdf.(Normal.(view(X, :,2:n_β+1)*β[i], σ[i]), view(X, :,1)) for i in 1:k])

    @inbounds for t in 1:T
        ξ[t,:] = t == 1 ? ξ_0 : view(ξ, t-1, :)
        #ξ_next = P'ξ[t, :]
        mul!(ξ_next, P, view(ξ, t, :))  # same as: ξ_next  = P*view(ξ, t, :)
        L[t] = view(η, t, :)'ξ_next
        @views @. ξ[t,:] = (1/L[t]) * ξ_next * η[t, :]
    end

    return (logsum ? sum(log.(L)) : L ), ξ #sum(log.(L)), ξ
end

function obj_func(θ, fΔ, x, k)
    
    if length(fΔ) > 0
        fΔ[1:length(θ)] .= FiniteDiff.finite_difference_gradient(θ -> -loglik(θ, x, k)[1], θ)
    end

    return -loglik(θ, x, k)[1]
end


function MSModel(y::Vector{Float64},
                 k::Int64, 
                 p::Int64,
                 ;exog_vars::Matrix{Float64}=Matrix{Float64}(undef, 0, 0),
                 x0::Vector{Float64}=Vector{Float64}(undef, 0),
                 algorithm::Symbol=:LN_SBPLX)

    @assert p >= 0 "Amount of lags shoould not be negative"
    @assert k >= 0 "Amount of states shoould not be negative"

    T   = size(y)[1] - p
    x   = add_lags(y, p)    
    x   = [x[:,1] ones(T) x[:, 2:end]]
    n_β = size(exog_vars)[2]+1+p

    if !isempty(exog_vars)
        @assert size(y)[1] == size(exog_vars)[1] "Number of observations is not the same between y and exog_vars"
        x = [x exog_vars[p+1:end, :]]
    end
    
    # also: LD_VAR2, :LD_VAR1, :LD_LBFGS, :LN_SBPLX
    opt               = Opt(algorithm, k + k*(size(x)[2]-1) + (k-1)*k)
    opt.lower_bounds  = [repeat([10e-10], k); repeat([-Inf], k*(size(x)[2]-1)); repeat([10e-10], (k-1)*k)]
    opt.xtol_rel      = 0
    opt.min_objective = (θ, fΔ) -> obj_func(θ, fΔ, x, k)
    
    if isempty(x0)
        
        kmeans_res = kmeans(reshape(x[:,1], 1, T), k)

        μ_em = kmeans_res.centers' 
        σ_em = [std(x[kmeans_res.assignments .== i, 1]) for i in 1:k]
        p_em = [sum(kmeans_res.assignments .== i ) / T for i in 1:k]

        # this is really bad code 
        # what i want to do is put the probabilites from kmeans into x0 anyhow
        pmat_em       = zeros(k,k)
        [pmat_em[i,i] = p_em[i] for i in 1:k]
        [pmat_em[i,j] = minimum(p_em) /2 for i in 1:k, j in 1:k if i != j]
        pmat_em       = pmat_em ./ sum(pmat_em, dims=1)
        pmat_em       = pmat_em[1:k-1, :] .* sum(pmat_em[1:k-1, :] .+ 1, dims=1) 
        p_em          = vec(pmat_em)

        μx0 = zeros(k*(size(x)[2]-1))
        μx0[1:n_β:n_β*k] .= μ_em[:] # kmeans centers are intercepts of the state equation  
        # μ_em = [μ_em[:][i] for i in 1:k for _ in 1:n_β] # alternatively: kmeans center is repeated n_β times

        x0 = [σ_em.^2; μx0; p_em]
        #x0 = [repeat([std(x[:,1])], k).^2; repeat([mean(x[:,1])], k*(size(x)[2]-1)); repeat([0.5],(k-1)*k)]
    end
    
    (minf,θ_hat,ret) = NLopt.optimize(opt, x0)
    
    println(ret)
    σ, β, P = trans_θ(θ_hat, k, n_β)
    rawP = θ_hat[(k+k*n_β+1):end]

    return MSM(β, σ, P, rawP, k, p, x, T, -minf)
end

function filtered_probs(msm_model::MSM, x::Matrix{Float64}=Matrix{Float64}(undef, 0, 0))
    
    if isempty(x)
        x = msm_model.x
    end

    θ_hat = [msm_model.σ; vcat(msm_model.β...); vec(msm_model.rawP)]
    ξ     = loglik(θ_hat, x, msm_model.k)[2]

    return ξ
end

function smoothed_probs(msm_model::MSM, x::Matrix{Float64}=Matrix{Float64}(undef, 0, 0))
    
    if isempty(x)
        x = msm_model.x
    end
    
    T = msm_model.T
    P = msm_model.P

    ξ        = filtered_probs(msm_model, x)
    ξ_T      = zeros(size(ξ))
    ξ_T[T,:] = ξ[T, :]

    for t in reverse(1:T-1)
        ξ_rate     = ξ_T[t+1, :] ./ (P*ξ[t, :])
        ξ_T[t, :] .= P' * ξ_rate .* ξ[t, :]
    end

    return ξ_T
end



k = 3
p = 0
# σ = sample(0.5:0.5:10, k)
# μ = sample(-2:0.5:5, k*(p+1))
# P = rand(1:10, k,k) ; P = P ./ sum(P, dims=1)
μ = [1.0, -0.5, 2.0] 
σ = [0.8,  1.5, 0.5] 
P = [0.7 0.2; 0.3 0.8]
P = [0.7 0.15 0.2; 0.2 0.75 0.15; 0.1 0.1 0.65] #[0.8 0.1; 0.2 0.9]  #
T = 200

θ = [sqrt.(σ); μ; vec(P[1:k*(k-1)])]

X, s_t = generate_mars(μ, σ, P, T+p, 0)

# using DelimitedFiles
# writedlm( "data/artificial.csv",  X, ',')

model = MSModel(X,k, p) 
model.β 
state_coeftable(model, 1)
model.P

summary(model)

transition_mat(model)

state_coeftable(model, 1)

using Plots

plot(smoothed_probs(model))

for color in [:red, :cyan, :blue, :magenta]
    printstyled("Hello World $(color)\n"; color = color)
end

