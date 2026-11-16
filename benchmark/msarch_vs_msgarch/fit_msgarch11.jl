# Fits a 2-regime MS-GARCH(1,1) model with MSGARCHModel() on weekly S&P 500 returns,
# for comparison against the equivalent fit from R's MSGARCH package (fit_msgarch11.R,
# same folder). See README.md for the reference numbers and how to read the result.
#
# Run from anywhere, e.g.:  julia benchmark/msarch_vs_msgarch/fit_msgarch11.jl

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)

using MarSwitching
using DelimitedFiles
using Random

data, header = readdlm(joinpath(@__DIR__, "spx_weekly_returns.csv"), ',', header = true)
y = Float64.(data[:, findfirst(==("spx"), vec(header))])

println("n obs: ", length(y))
println("mean(y) = ", sum(y) / length(y), "   sd(y) = ", sqrt(sum((y .- sum(y) / length(y)) .^ 2) / (length(y) - 1)))

# k = 2 regimes, GARCH(1,1), zero mean (intercept = "no") to match MSGARCH's sGARCH
# model, which has no mean/location submodel. switching_var = true (default)
# lets omega, alpha and gamma (MSGARCH's "beta") all differ across regimes.
Random.seed!(42)
t0 = time()
model = MSGARCHModel(y, 2, 1, 1,
                     intercept = "no",
                     switching_var = true,
                     random_search_em = 8,
                     random_search = 8,
                     verbose = false)

println("\n=== MSGARCHModel fit ===")
println("fit time    = ", round(time() - t0, digits = 1), " s")
println("nlopt_msg   = ", model.nlopt_msg)
println("omega       = ", model.ω)
println("alpha       = ", [model.α[s][1] for s in 1:2])
println("gamma       = ", [model.γ[s][1] for s in 1:2], "   (MSGARCH's beta)")
println("P (diag)    = ", (model.P[1, 1], model.P[2, 2]))
println("expected duration (weeks) = ", (1 / (1 - model.P[1, 1]), 1 / (1 - model.P[2, 2])))
println("Likelihood  = ", model.Likelihood)

summary_msm(model)
