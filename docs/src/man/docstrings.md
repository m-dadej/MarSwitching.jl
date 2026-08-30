```@meta
Description = "API reference for MarSwitching.jl: docstrings for MSModel, MSARCHModel, generate_msm, filtered_probs, smoothed_probs, expected_duration, predict, transition_mat and every exported function."
```

# API reference
!!! tip "You may also use help mode `?`"
    You can access the docstring of every function listed here by typing `?` in Julia's REPL followed by the function name. For example, `?MSModel` will show the docstring for the `MSModel` function.

MarSwitching.jl exports following list of functions (and a struct):
```@index
```

## Model estimation
```@docs
MarSwitching.MSM
MarSwitching.MSModel
MarSwitching.MSARCHModel
MarSwitching.grid_search_msm
```

## Simulation 
```@docs
MarSwitching.generate_msm
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
MarSwitching.conditional_variance
```
## Other
```@docs
MarSwitching.add_lags
```