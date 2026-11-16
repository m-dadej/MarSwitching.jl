# Fits a 2-regime MS-GARCH(1,1) with MSGARCHModel() to simulated data with known
# parameters (sim_msgarch11_data.csv, from sim_msgarch11_generate.jl) and compares the
# estimates with the truth. sim_msgarch11_fit.R does the same with R's MSGARCH package,
# printing an identically laid-out table. See README.md for both sets of results.
#
# Run from anywhere, e.g.:  julia benchmark/msarch_vs_msgarch/sim_msgarch11_fit.jl

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)

using MarSwitching
using DelimitedFiles
using FiniteDiff
using LinearAlgebra
using Printf
using Random

const M = MarSwitching

data, _   = readdlm(joinpath(@__DIR__, "sim_msgarch11_data.csv"), ',', header = true)
y         = Float64.(data[:, 2])
s_true    = Int.(data[:, 3])
T         = length(y)
pars, _   = readdlm(joinpath(@__DIR__, "sim_msgarch11_params.csv"), ',', header = true)
names     = String.(pars[:, 1])
truth     = Float64.(pars[:, 2])

Random.seed!(42)
t_fit = @elapsed model = MSGARCHModel(y, 2, 1, 1,
                                      intercept = "no",
                                      random_search_em = 2,
                                      random_search = 2,
                                      verbose = false)

# Estimates in MSGARCH's parameter order and naming, with regimes sorted by unconditional
# variance ω/(1-α-γ) so that regime 1 is the calm one, as in the data-generating process.
uncond = [model.ω[s] / (1 - model.α[s][1] - model.γ[s][1]) for s in 1:2]
a, b   = sortperm(uncond)
function natural(θ)
    ω, α, γ, _, P, _ = M.trans_θ_garch(θ, 2, 0, 0, "no", true, false, :normal, 1, 1)
    # P is left-stochastic, P[i, j] = P(S_t = i | S_{t-1} = j), so MSGARCH's P_2_1 is P[a, b]
    return [ω[a], α[a][1], γ[a][1], ω[b], α[b][1], γ[b][1], P[a, a], P[a, b]]
end
est = natural(model.raw_params)

# delta-method standard errors from the inverse Hessian of the raw parameters
# (raw_vcov is internal; summary_msm() prints the same SEs for ω, α, γ)
J  = FiniteDiff.finite_difference_jacobian(natural, model.raw_params)
se = sqrt.(abs.(diag(J * M.raw_vcov(model) * J')))

# Log-likelihood as MSGARCH reports it: each regime's variance starts at its unconditional
# level, y_1 enters only the variance recursion, and the filter starts at t = 2 from the
# ergodic probabilities. (MarSwitching backcasts h with the mean squared residual and
# includes t = 1.) Reproduces MSGARCH's reported log-likelihood to all printed digits.
function loglik_msgarch_convention(y, p)
    ω, α, β = [p[1], p[4]], [p[2], p[5]], [p[3], p[6]]
    P = [p[7] p[8]; 1-p[7] 1-p[8]]
    h = [ω[s] / (1 - α[s] - β[s]) for s in 1:2]
    ξ = ergodic_probs(P)
    ll = 0.0
    for t in 2:length(y)
        h  = ω .+ α .* y[t-1]^2 .+ β .* h
        pr = P * ξ
        η  = exp.(-y[t]^2 ./ (2 .* h)) ./ sqrt.(2π .* h)
        L  = dot(η, pr)
        ll += log(L)
        ξ  = pr .* η ./ L
    end
    return ll
end

# the true parameters in MarSwitching's raw layout: ω, α, γ blocks, then P (one raw per column)
r1, r2  = M.garch_raw(truth[2:3]), M.garch_raw(truth[5:6])
θ_true  = [sqrt(truth[1]); sqrt(truth[4]); r1[1]; r2[1]; r1[2]; r2[2];
           truth[7] / (1 - truth[7]); truth[8] / (1 - truth[8])]
ll_true = M.loglik(θ_true, model.x, 2, 0, 0, "no", true, :normal, 1, 1)[1]

# share of observations whose most likely smoothed regime is the true one
sp  = smoothed_probs(model)[:, [a, b]]
acc = sum((sp[:, 2] .> 0.5) .== (s_true .== 2)) / T

println("=== MSGARCHModel (MarSwitching.jl) on simulated MS-GARCH(1,1), T = $T ===")
@printf "fit time: %.1f s   nlopt_msg: %s\n\n" t_fit model.nlopt_msg
@printf "%-10s %10s %10s %10s %12s\n" "parameter" "true" "estimate" "std.err" "(est-true)/se"
for i in eachindex(names)
    @printf "%-10s %10.4f %10.4f %10.4f %12.2f\n" names[i] truth[i] est[i] se[i] (est[i] - truth[i]) / se[i]
end
println()
@printf "log-likelihood, MarSwitching convention:  estimate %.4f | truth %.4f\n" model.Likelihood ll_true
@printf "log-likelihood, MSGARCH convention:       estimate %.4f | truth %.4f\n" loglik_msgarch_convention(y, est) loglik_msgarch_convention(y, truth)
@printf "regimes correctly classified (smoothed probabilities): %.1f%%\n" 100 * acc
