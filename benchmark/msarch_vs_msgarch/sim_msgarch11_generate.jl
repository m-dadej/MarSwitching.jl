# Simulates the dataset used by sim_msgarch11_fit.jl / sim_msgarch11_fit.R: a 2-regime,
# zero-mean MS-GARCH(1,1) process (Haas, Mittnik & Paolella, 2004) with known parameters,
# chosen so that every parameter is clearly non-zero, away from its bounds, and precisely
# estimable: at T = 10,000 the asymptotic standard errors are about 3-11% of the true
# α and γ and 15-17% of ω (GARCH intercepts are imprecise in any sample of realistic size).
# Both regimes are equally likely and last 100 periods on average, so each is well informed.
#
# Writes sim_msgarch11_data.csv (t, y, true regime) and sim_msgarch11_params.csv (true
# parameters, named as in R's MSGARCH). Both CSVs are committed, so the fit scripts do not
# depend on this script: Julia does not guarantee identical random streams across
# versions, so re-running it may produce a different (equally valid) sample.
#
# Run from anywhere, e.g.:  julia benchmark/msarch_vs_msgarch/sim_msgarch11_generate.jl

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)

using MarSwitching
using DelimitedFiles
using Random

# regime 1: calm        unconditional variance 0.10 / (1 - 0.15 - 0.75) = 1.0
# regime 2: turbulent   unconditional variance 1.00 / (1 - 0.25 - 0.60) = 6.67
ω = [0.10, 1.00]
α = [[0.15], [0.25]]
γ = [[0.75], [0.60]]               # MSGARCH calls these beta
# left-stochastic: P[i, j] = P(S_t = i | S_{t-1} = j). Expected durations 100 and 100
P = [0.99 0.01;
     0.01 0.99]
T = 10_000

Random.seed!(2026)
y, s_t, _ = generate_msm([0.0, 0.0], ω, α, γ, P, T)

writedlm(joinpath(@__DIR__, "sim_msgarch11_data.csv"),
         [["t" "y" "regime"]; hcat(Any[1:T...], y, s_t)], ',')   # Any: keep t, regime as integers

# MSGARCH naming; P_2_1 = P(S_t = 1 | S_{t-1} = 2), i.e. P[1, 2] above
params = ["alpha0_1" ω[1]; "alpha1_1" α[1][1]; "beta_1" γ[1][1];
          "alpha0_2" ω[2]; "alpha1_2" α[2][1]; "beta_2" γ[2][1];
          "P_1_1" P[1, 1]; "P_2_1" P[1, 2]]
writedlm(joinpath(@__DIR__, "sim_msgarch11_params.csv"), [["name" "value"]; params], ',')

println("T = ", T, ", share of regime 2 = ", round(sum(s_t .== 2) / T, digits = 3),
        ", sd(y) by regime = ", [round(sqrt(sum(y[s_t .== s] .^ 2) / sum(s_t .== s)), digits = 3) for s in 1:2])
