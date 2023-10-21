module MarSwitching

using FiniteDiff
using LinearAlgebra
using NLopt

import Distributions: Normal, pdf, Chi, cdf, Uniform
import Random: rand
import StatsBase: Weights, sample, std, mean
import Printf: @printf

include("msmodel.jl")
include("likelihood.jl")
include("generate.jl")
include("inference.jl")
include("utils.jl")
include("results.jl")

export generate_msm, MSModel, filtered_probs, smoothed_probs
export get_std_errors, expected_duration, state_coeftable, transition_mat, summary_msm
export MSM, add_lags, ergodic_probs
export predict, coeftable_tvtp

end
