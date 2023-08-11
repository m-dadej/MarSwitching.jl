module MARS

using Distributions
using LinearAlgebra
using NLopt
using Random
using StatsBase
using FiniteDiff
using LineSearches
using Clustering
using Printf


include("likelihood.jl")
include("helpers.jl")
include("results.jl")
# Write your package code here.

export generate_mars, MSM, loglik, MSModel, filtered_probs, smoothed_probs, add_lags
export get_std_errors, expected_duration, state_coeftable, transition_mat, summary_mars
export loglik

end
