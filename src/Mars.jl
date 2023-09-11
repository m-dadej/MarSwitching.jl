module Mars

using Distributions
using LinearAlgebra
using NLopt
using Random
using StatsBase
using FiniteDiff
using LineSearches
using Clustering
using Printf

include("msmodel.jl")
include("likelihood.jl")
include("generate.jl")
include("inference.jl")
include("utils.jl")
include("results.jl")
# Write your package code here.

export generate_mars,  MSModel, filtered_probs, smoothed_probs
export get_std_errors, expected_duration, state_coeftable, transition_mat, summary_mars
export loglik, MSM, add_lags
export predict, loglik_tvtp, coeftable_tvtp

end
