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


include("likelihood.jl")
include("helpers.jl")
include("results.jl")
# Write your package code here.

export generate_mars,  MSModel, filtered_probs, smoothed_probs
export get_std_errors, expected_duration, state_coeftable, transition_mat, summary_mars

# delete after development
export loglik, MSM, add_lags, vec2param, trans_θ, obj_func
export vec2param_nointercept, vec2param_nonswitch, vec2param_switch
export predict, P_tvtp, obj_func_tvtp, loglik_tvtp

end
