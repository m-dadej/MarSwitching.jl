using Random

# how to replicate the error
# due to numerical instability the steady states are negative (e.g.  -1.7763568394002505e-15)
# usualy when the state is very unlikely given the initial P

# new version of the likelihood will have some checks for this:
# maybe :
# ξ_0 = any(ξ_0 .< 0) ? ones(k) ./ k : ξ_0

k = 3
μ = [1.0, -0.5, 0.12] 
β_ns = Vector{Float64}([0.633])
σ = [0.2, 0.2, 0.2] 
P = [0.9 0.05 0.1; 0.05 0.85 0.05; 0.05 0.1 0.85]
T = 100

Random.seed!(1)
y, s_t, X = generate_mars(μ, σ, P, T, β_ns = β_ns)

model = MSModel(y, k,   exog_vars = reshape(X[:,2],T,1),
                        switching_var = false,
                        maxtime = 100000000)




θ = [0.6034022872782692, -0.33468866364099453, -0.10077949646321205, 0.9373199318382164, 0.5754121761202129, 9.559282551113908, 0.5449864362027925, 0.0, 5.023052566524919, 0.0, 0.0]
k = model.k
n_β = model.n_β
n_β_ns = model.n_β_ns
intercept = model.intercept
switching_var = model.switching_var

σ, β, P = trans_θ(θ, k, n_β, n_β_ns, intercept, switching_var, false)


X = [y X[:,end-1:end]]
T      = size(X)[1]
ξ      = zeros(T, k)  # unconditional transition probabilities at t
L      = zeros(T)     # likelihood 
ξ_next = zeros(k)     # unconditional transition probabilities at t+1

k = model.k
n_β = model.n_β
n_β_ns = model.n_β_ns
intercept = model.intercept
switching_var = model.switching_var
θ = params

σ, β, P = trans_θ(θ, k, n_β, n_β_ns, intercept, switching_var, false)

# if tvtp
#     P = trans_tvtp(P) 
# end

#initial guess for the unconditional probabilities

A = [I - P; ones(k)']

# check if A'A is invertible
ξ_0 = !isapprox(det(A'A), 0) ? (inv(A'A)*A')[:,end] : ones(k) ./ k

# 3-element Vector{Float64}:
#   3.552713678800501e-15
#  -1.7763568394002505e-15
#   1.0