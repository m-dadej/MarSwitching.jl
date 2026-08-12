# Fits a 2-regime MS-ARCH(1) model with MSARCHModel() on weekly S&P 500 returns,
# for comparison against the equivalent fit from R's MSGARCH package (fit_msgarch.R,
# same folder). See README.md for the reference numbers and how to read the result.
#
# Run from anywhere, e.g.:  julia benchmark/msarch_vs_msgarch/fit_msarch.jl

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)

using MarSwitching
using DelimitedFiles
using Random

data, header = readdlm(joinpath(@__DIR__, "spx_weekly_returns.csv"), ',', header = true)
y = Float64.(data[:, findfirst(==("spx"), vec(header))])

println("n obs: ", length(y))
println("mean(y) = ", sum(y) / length(y), "   sd(y) = ", sqrt(sum((y .- sum(y) / length(y)) .^ 2) / (length(y) - 1)))

# k = 2 regimes, ARCH(1), zero mean (intercept = "no") to match MSGARCH's sARCH
# model, which has no mean/location submodel. switching_var = true (default)
# lets both omega and alpha differ across regimes.
Random.seed!(42)
model = MSARCHModel(y, 2, 1,
                    intercept = "no",
                    switching_var = true,
                    random_search_em = 8,
                    random_search = 8,
                    verbose = false)

println("\n=== MSARCHModel fit ===")
println("nlopt_msg   = ", model.nlopt_msg)
println("omega       = ", model.ω)
println("alpha       = ", [model.α[s][1] for s in 1:2])
println("P (diag)    = ", (model.P[1, 1], model.P[2, 2]))
println("expected duration (weeks) = ", (1 / (1 - model.P[1, 1]), 1 / (1 - model.P[2, 2])))
println("Likelihood  = ", model.Likelihood)

summary_msm(model)
