module Mars

import Distributions: Normal, pdf, Chi, cdf, Uniform
using LinearAlgebra
using NLopt
import Random: rand
import StatsBase: Weights, sample, std, mean
using FiniteDiff
import Clustering: kmeans
import Printf: @printf

include("msmodel.jl")
include("likelihood.jl")
include("generate.jl")
include("inference.jl")
include("utils.jl")
include("results.jl")

export generate_mars,  MSModel, filtered_probs, smoothed_probs
export get_std_errors, expected_duration, state_coeftable, transition_mat, summary_mars
export MSM, add_lags
export predict, coeftable_tvtp

end
