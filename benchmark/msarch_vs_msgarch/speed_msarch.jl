# Speed benchmark for MSARCHModel() on the same 2-regime ARCH(1) fit used in
# fit_msarch.jl / fit_msgarch.R (see README.md, "Speed comparison" section).
#
# Reports (a) the first-call ("cold") time, which includes one-time JIT
# compilation, and (b) steady-state ("warm") timing over repeated calls at
# default effort (random_search_em = 0, random_search = 0 -- a single EM
# initialization followed by a single NLopt optimization run), to match
# MSGARCH's FitML() default of one optimization run from one starting point.
#
# Run from anywhere, e.g.:  julia benchmark/msarch_vs_msgarch/speed_msarch.jl

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)

using MarSwitching
using DelimitedFiles
using Statistics
using Random

data, header = readdlm(joinpath(@__DIR__, "spx_weekly_returns.csv"), ',', header = true)
y = Float64.(data[:, findfirst(==("spx"), vec(header))])

fit() = MSARCHModel(y, 2, 1, intercept = "no", verbose = false)

Random.seed!(42)
t_cold = @elapsed model = fit()
println("cold call (incl. JIT compilation): ", round(t_cold, digits = 3), " s")
println("  Likelihood = ", model.Likelihood, "   nlopt_msg = ", model.nlopt_msg)

N = 20
times = Float64[]
Random.seed!(42)
for _ in 1:N
    t = @elapsed fit()
    push!(times, t)
end

println("\nwarm calls, N = ", N, " (default effort: 1 EM init + 1 NLopt run)")
println("  mean   = ", round(mean(times), digits = 4), " s")
println("  median = ", round(median(times), digits = 4), " s")
println("  min    = ", round(minimum(times), digits = 4), " s")
println("  max    = ", round(maximum(times), digits = 4), " s")
