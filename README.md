## MARS.jl: MARkov Switching models in Julia

[![Build Status](https://github.com/m-dadej/MARS.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/m-dadej/MARS.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Build Status](https://app.travis-ci.com/m-dadej/MARS.jl.svg?branch=main)](https://app.travis-ci.com/m-dadej/MARS.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/2o7c7dny19u0e18u?svg=true)](https://ci.appveyor.com/project/m-dadej/mars-jl-ovb60)
[![Coverage](https://codecov.io/gh/m-dadej/MARS.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/m-dadej/MARS.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Mars is a package for estimation of Markov switching dynamic models in Julia. The package is currently being developed, altough the basic functionality is already available. 

Contact: Mateusz Dadej, m.dadej at unibs.it


## Installation
```julia
Pkg.add("https://github.com/m-dadej/Mars.jl")
# or
] add https://github.com/m-dadej/Mars.jl
```

## Example
```julia
using Mars

k = 3                   # number of regimes
T = 1000                # number of generated observations
μ = [1.0, -0.5, 0.12]   # regime-switching intercepts
β = [-1.5, 0.9, 0.0]    # regime-switching coefficient for β
σ = [0.4,  0.5, 0.2]    # regime-switching standard deviation
P = [0.9 0.05 0.1       # transition matrix (left-stochastic)
     0.05 0.85 0.05
     0.05 0.1 0.85]

Random.seed!(1)
# generate artificial data with given parameters
y, s_t, X = generate_mars(μ, σ, P, T, β = β) 

# estimate the model
model = MSModel(y, k, intercept = "switching", exog_switching_vars = reshape(X[:,2],T,1))

# output summary table
summary_mars(model)
````

The 'summary_mars(model)' will output following summary table:

```jldoctest
Markov Switching Model with 3 regimes
=====================================================
# of observations:         1000 Loglikelihood:            -717.263 
# of estimated parameters:   15  AIC                      1464.526 
Error distribution:    Gaussian  BIC                      1538.142 
------------------------------------------------------
------------------------------
Summary of regime 1: 
------------------------------
Coefficient  |  Estimate  |  Std. Error  |  z value  |  Pr(>|z|)   
-------------------------------------------------------------------
β_0          |     0.107  |       0.013  |    8.231  |    < 1e-3  
β_1          |    -0.014  |       0.013  |   -1.077  |     0.281   
σ            |     0.197  |       0.011  |   17.909  |    < 1e-3   
-------------------------------------------------------------------
Expected regime duration: 7.18 periods
-------------------------------------------------------------------
------------------------------
Summary of regime 2: 
------------------------------
Coefficient  |  Estimate  |  Std. Error  |  z value  |  Pr(>|z|)   
-------------------------------------------------------------------
β_0          |    -0.486  |       0.033  |  -14.727  |    < 1e-3  
β_1          |     0.935  |       0.032  |   29.219  |    < 1e-3  
σ            |     0.498  |       0.017  |   29.294  |    < 1e-3
-------------------------------------------------------------------
Expected regime duration: 6.69 periods
-------------------------------------------------------------------
------------------------------
Summary of regime 3:
------------------------------
Coefficient  |  Estimate  |  Std. Error  |  z value  |  Pr(>|z|)
-------------------------------------------------------------------
β_0          |       1.0  |        0.02  |     50.0  |    < 1e-3  
β_1          |    -1.497  |       0.022  |  -68.045  |    < 1e-3  
σ            |      0.41  |       0.011  |   37.273  |    < 1e-3
-------------------------------------------------------------------
Expected regime duration: 12.58 periods
-------------------------------------------------------------------
left-stochastic transition matrix:
          | regime 1   | regime 2   | regime 3
----------------------------------------------------
 regime 1 |   86.064%  |    9.945%  |    2.519%  |
 regime 2 |    5.111%  |   85.046%  |     5.43%  |
 regime 3 |    8.825%  |    5.009%  |   92.051%  |

 ```
