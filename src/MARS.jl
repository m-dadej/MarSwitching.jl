module MARS

include("likelihood.jl")
include("helpers.jl")
include("results.jl")
# Write your package code here.

export generate_mars, MSM, loglik, MSModel, filtered_probs, smoothed_probs
export get_std_errors, expected_duration, state_coeftable, transition_mat, summary

end
