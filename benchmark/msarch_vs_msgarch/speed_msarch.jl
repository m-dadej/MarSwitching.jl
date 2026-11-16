# Speed benchmark for MSARCHModel() on the same 2-regime ARCH(1) fit used in
# fit_msarch.jl / fit_msgarch.R (see README.md, "Speed comparison" section).
#
# Reports (a) the first-call ("cold") time, which includes one-time JIT compilation, and
# (b) steady-state ("warm") timing over repeated calls for four variants:
#   - the defaults (:LN_SBPLX) and algorithm = :LD_LBFGS,
#   - each with a random EM starting point per call (the default: the EM start is drawn
#     from the global RNG) or with a fixed one (RNG re-seeded before every call, so every
#     call starts from the same EM point).
# Default effort (random_search_em = 0, random_search = 0) is one EM initialization followed
# by 5 NLopt runs: one from the EM start plus MSModel()'s 4 structured ARCH restarts (over
# total ARCH persistence). MSGARCH's FitML() makes one optimization run from one start.
#
# Run from anywhere, e.g.:  julia benchmark/msarch_vs_msgarch/speed_msarch.jl

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)

using MarSwitching
using DelimitedFiles
using Printf
using Statistics
using Random

data, header = readdlm(joinpath(@__DIR__, "spx_weekly_returns.csv"), ',', header = true)
y = Float64.(data[:, findfirst(==("spx"), vec(header))])

fit(alg) = MSARCHModel(y, 2, 1, intercept = "no", algorithm = alg, verbose = false)

Random.seed!(42)
t_cold = @elapsed model = fit(:LN_SBPLX)
@printf "cold call, defaults (incl. JIT compilation): %.2f s   Likelihood = %.6f\n\n" t_cold model.Likelihood

N = 20
@printf "warm calls, N = %d\n" N
ll_best = model.Likelihood   # 4277.1046, the optimum MSGARCH also reaches (see README.md)
@printf "%-36s %8s %8s %8s %8s   %-28s %s\n" "variant" "mean" "median" "min" "max" "log-likelihood range" "at best optimum"
for (label, alg, fixed) in (("defaults (:LN_SBPLX), random start", :LN_SBPLX, false),
                            ("defaults (:LN_SBPLX), fixed start",  :LN_SBPLX, true),
                            (":LD_LBFGS, random start",            :LD_LBFGS, false),
                            (":LD_LBFGS, fixed start",             :LD_LBFGS, true))
    fit(alg)                                         # compile this variant's code path
    times, lls = Float64[], Float64[]
    Random.seed!(42)
    for _ in 1:N
        fixed && Random.seed!(42)
        t = @elapsed m = fit(alg)
        push!(times, t); push!(lls, m.Likelihood)
    end
    @printf "%-36s %7.2fs %7.2fs %7.2fs %7.2fs   %.4f to %.4f   %d/%d\n" label mean(times) median(times) minimum(times) maximum(times) minimum(lls) maximum(lls) sum(lls .> ll_best - 1e-3) N
end
