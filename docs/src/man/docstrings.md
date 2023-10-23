# Docstrings
!!! tip "You may also use help mode `?`"
    You can access the docstring of every function listed here by typing `?` in Julia's REPL followed by the function name. For example, `?MSModel` will show the docstring for the `MSModel` function.

MarSwitching.jl exports following list of functions (and a struct):
```@index
```

## Model estimation
```@docs
MarSwitching.MSM
MarSwitching.MSModel
MarSwitching.grid_search_msm
```

## Simulation 
```@docs
MarSwitching.generate_msm(model::MSM, T::Int64 = 0)
MarSwitching.generate_msm(μ::Vector{V},
                        σ::Vector{V},
                        P::Matrix{V},
                        T::Int64;
                        β::Vector{V} = Vector{V}([]),
                        β_ns::Vector{V} = Vector{V}([]),
                        δ::Vector{V} = Vector{V}([]),
                        tvtp_intercept::Bool = true) where V <: AbstractFloat
```

## Model summary
```@docs
MarSwitching.summary_msm
MarSwitching.transition_mat
MarSwitching.state_coeftable
MarSwitching.coeftable_tvtp
MarSwitching.get_std_errors
```
## Model inference
```@docs
MarSwitching.filtered_probs
MarSwitching.smoothed_probs
MarSwitching.predict
MarSwitching.expected_duration
MarSwitching.ergodic_probs
```
## Other
```@docs
MarSwitching.add_lags
```