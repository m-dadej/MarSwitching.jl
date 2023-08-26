## Mars.jl: MARkov Switching dynamic models in Julia

[![Build Status](https://github.com/m-dadej/MARS.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/m-dadej/MARS.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Build Status](https://app.travis-ci.com/m-dadej/MARS.jl.svg?branch=main)](https://app.travis-ci.com/m-dadej/MARS.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/2o7c7dny19u0e18u?svg=true)](https://ci.appveyor.com/project/m-dadej/mars-jl-ovb60)
[![Coverage](https://codecov.io/gh/m-dadej/MARS.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/m-dadej/MARS.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Mars.jl is a package for estimating Markov switching dynamic models (also called regime switching) in Julia. The package is currently being developed, altough the basic functionality is already available. 

Contact: Mateusz Dadej, m.dadej at unibs.it


## Installation
```julia
Pkg.add("https://github.com/m-dadej/Mars.jl")
# or
] add https://github.com/m-dadej/Mars.jl
```
## Markov regime switching model in a nutshell

The markov switching models are a class of models that allow for the parameters to change over time, depending on the unobservable state like economic recession, high volatility on financial markets or epidemiologic outbreak. The state follows markov process with a given probability transition matrix for each of $k$ states:

```math
\begin{equation*}
P(S_t = i | S_{t-1} = j) = 
\begin{pmatrix}
p_{1,1} & p_{1,2} & \cdots & p_{1,k} \\
p_{2,1} & p_{2,2} & \cdots & p_{2,k} \\
\vdots  & \vdots  & \ddots & \vdots  \\
p_{k,1} & p_{k,2} & \cdots & p_{k,k} 
\end{pmatrix}
\end{equation*}
```

Satisfying standard markovian properties. The general model is defined as follows:

```math
\begin{align*}
\mathbf{y}_t &= \mathbf{\mu}_S + \mathbf{\beta}_{S}' \mathbf{X}_t + \mathbf{\delta}'\mathbf{Z}_t + \mathbf{\epsilon}_t, & \mathbf{\epsilon} \sim f(0,\mathcal{\Sigma}_s)\\
\end{align*}
```

Where $\mathbf{y}_t$ is vector of dependent variables, $\mathbf{\mu}_s$ and $\mathbf{\beta}_s$ are model parameters dependent on the state $S_t$, $\mathbf{\delta}$ is a vector of parameters for exogenous variables. The error is distributed according to some distribution $f$ with state dependent covariance matrix $\mathcal{\Sigma}_s$. 

Because of the unobserved nature of the state, the model is estimated by maximum likelihood. The likelihood function is calculated using the method described in Hamilton, 1989.

## Functionality 

- Currently available:
    - Markov switching model with $k$ regimes and combinations of:
        - switching/non-switching or without intercept
        - switching/non-switching variance
        - switching/non-switching exogenous variables
    - Filtered probabilites
    - Smoothed probabilites (Kim, 1994)
    - Summary statistics of coefficients
    - in-sample and out-of-sample `predict()`
    - Expected regime duration
    - Simulation of data from Markov switching model with:
        - switching/non-switching or without intercept
        - switching variance
        - switching/non-switching exogenous variables
    - Adding lagged variables to the matrix
- Planned functionality:
    - in-sample and out-of-sample predict() function
    - simulating data from estimated model
    - other error distributions (t, skew-t, etc.)
    - variable and number of states selection
    - time-varying transition probabilites (Filardo 1994)
    - Markov Switching GARCH model
    - Markov Switching VAR model
    - Markov Switching model with lagged states. E.g. $y_t = \mu_{S_t} + \phi(y_{t-1} - \mu_{S_{t-1}})$
    
## 

## Example

Following example will estimate a simple Markov switching model with regime dependent intercept, exogenous variable and variance. The model is defined as follows:

```math
\begin{align*}
    y_t &= \mu_s + \beta_s x_t + \epsilon_t, & \epsilon &\sim \mathbb{N}(0,\mathcal{\Sigma}_s) \\
\end{align*}
```
```math
\begin{equation*}
    P(S_t = i | S_{t-1} = j) = \begin{bmatrix}
        p_1 & 1 - p_2\\
        1 - p_1 & p_2
        \end{bmatrix}
\end{equation*}
```

```julia
using Mars

k = 2            # number of regimes
T = 400          # number of generated observations
μ = [1.0, -0.5]  # regime-switching intercepts
β = [-1.5, 0.0]  # regime-switching coefficient for β
σ = [1.1,  0.8]  # regime-switching standard deviation
P = [0.9 0.05    # transition matrix (left-stochastic)
     0.1 0.95]

Random.seed!(123)

# generate artificial data with given parameters
y, s_t, X = generate_mars(μ, σ, P, T, β = β) 

# estimate the model
model = MSModel(y, k, intercept = "switching", exog_switching_vars = reshape(X[:,2],T,1))

# output summary table
summary_mars(model)
````

The `summary_mars(model)`  will output following summary table:

```jldoctest
Markov Switching Model with 2 regimes
=====================================================
# of observations:          400 Loglikelihood:            -576.692 
# of estimated parameters:    8  AIC                      1169.384 
Error distribution:    Gaussian  BIC                      1201.316 
------------------------------------------------------
------------------------------
Summary of regime 1:
------------------------------
Coefficient  |  Estimate  |  Std. Error  |  z value  |  Pr(>|z|)   
-------------------------------------------------------------------
β_0          |     0.824  |       0.132  |    6.242  |    < 1e-3  
β_1          |    -1.483  |        0.12  |  -12.358  |    < 1e-3   
σ            |     1.124  |       0.046  |   24.435  |    < 1e-3   
-------------------------------------------------------------------
Expected regime duration: 11.34 periods
-------------------------------------------------------------------
------------------------------
Summary of regime 2:
------------------------------
Coefficient  |  Estimate  |  Std. Error  |  z value  |  Pr(>|z|)
-------------------------------------------------------------------
β_0          |    -0.516  |       0.052  |   -9.923  |    < 1e-3  
β_1          |    -0.003  |       0.051  |   -0.059  |     0.953  
σ            |     0.843  |       0.022  |   38.318  |    < 1e-3
-------------------------------------------------------------------
Expected regime duration: 28.58 periods
-------------------------------------------------------------------
left-stochastic transition matrix:
          | regime 1   | regime 2
---------------------------------------
 regime 1 |   91.181%  |    3.499%  |
 regime 2 |    8.819%  |   96.501%  |
 ```

The package also provides a function for filtered transition probabilites $P(S_t = i | \Psi_t)$, as well as smoothed ones (Kim, 1994) $P(S_t = i | \Psi_T)$. Essentially, the difference is that, in order to calculate the smoothed probabilites the whole sample is used.

```julia
using Plots

plot(smoothed_probs(model),
     label=["Regime 1" "Regime 2"],
     title = "Smoothed transition robabilities", 
     linewidth=2)
```     
![Plot](img/transition_probs.svg)

```julia
plot([smoothed_probs(model)[:,2] s_t.-1],
     label=["Regime 1" "Actual Regime"],
     title = "Smoothed transition robabilities",
     linewidth=2)  
```
 ![Plot](img/actual_probs.svg)

## Functions

The function for estimating the markov switching model is: 

```julia
MSModel(y::Vector{Float64},                     # vector of dependent variable
        k::Int64,                               # number of regimes
        ;intercept::String,                     # "switching" (default), "non-switching" or "no" intercept
        exog_vars::Matrix{Float64}              # optional matrix of exogenous variables
        exog_switching_vars::Matrix{Float64},   # optional matrix of exogenous variables with regime switching
        switching_var::Bool = true,             # is variance state-dependent?
        x0::Vector{Float64},                    # optional initial values of parameters for optimization
        algorithm::Symbol,                      # optional algorithm for NLopt.jl
        maxtime::Int64)                         # optional maximum time for optimization
```
The function returns `MSM` type object:

```julia
struct MSM 
    β::Vector{Vector{Float64}}  # β[state][i] vector of β for each state
    σ::Vector{Float64}          # error variance
    P::Matrix{Float64}          # transition matrix
    k::Int64                    # number of regimes
    n_β::Int64                  # number of β parameters
    n_β_ns::Int64               # number of non-switching β parameters
    intercept::String           # "switching" or "non-switching"
    switching_var::Bool         # is variance state-dependent?
    x::Matrix{Float64}          # data matrix
    T::Int64                    # number of observations
    Likelihood::Float64         # log-likelihood
    raw_params::Vector{Float64} # vector of parameters for optimization
end
```  
Filtered transition probabilites can be calculated from estimated model:

```julia
filtered_probs(msm_model::MSM,                       # estimated model
               y::Vector{Float64},                   # optional vector of dependent variables
               exog_vars::Matrix{Float64}            # optional matrix of exogenous variables
               exog_switching_vars::Matrix{Float64}) # optional matrix of exogenous variables with regime switching
                
```

Similarily, smoothed transition probabilites can be also calculated from estimated model:

```julia
smoothed_probs(msm_model::MSM,                       # estimated model
               y::Vector{Float64},                   # optional vector of dependent variables
               exog_vars::Matrix{Float64}            # optional matrix of exogenous variables
               exog_switching_vars::Matrix{Float64}) # optional matrix of exogenous variables with regime switching
```

The `predict()` function can be used to calculate instanteous or one step ahead predictions from estimated model:

```julia
predict(model::MSM,                             # estimated model
        instanteous::Bool = false;              # instanteous or one-step ahead prediction
        y::Vector{Float64},                     # optional vector of dependent variables
        exog_vars::Matrix{Float64},             # optional matrix of exogenous variables
        exog_switching_vars::Matrix{Float64})   # optional matrix of exogenous variables with regime switching
    
```
Which is the probability weighted average of predictions of each state equation:
    
```math
\hat{y}_t = \sum_{i=1}^{k} \hat{\xi}_{i,t}X_{t}'\hat{\beta}_{i}
```
And for one step ahead, the state probabilities have to be predicted themselves:

```math
\hat{y}_{t+1} = \sum_{i=1}^{k} (P\hat{\xi}_{i,t})X_{t+1}'\hat{\beta}_{i}
```

The provided new data needs to match the data used for estimation (with except of observations size). If not provided the prediction is done on the data used for estimation. For one step ahed forecast, there is no look-ahead bias, the y vector needs to be provided in order to calculate the state probabilites at time $t$.


The `summary_mars(model::MSM; digits::Int64=3)` function outputs a summary table that is built from 2 functions: 
- `transition_mat(model::MSM; digits::Int64=2)` - prints transition matrix
- `state_coeftable(model::MSM, state::Int64; digits::Int64=3)` - prints coefficient table for given state

It is also possible to simulate data from a given parameters

```julia
generate_mars(μ::Vector{Float64},    # vector of intercepts for each state
              σ::Vector{Float64},    # vector of error variances for each state
              P::Matrix{Float64},    # transition matrix
              T::Int64;              # number of observations
              β::Vector{Float64},    # vector of coefficients for each state
              β_ns::Vector{Float64}) # vector of non-switching coefficients
```

The function returns a tuple of 3 elements, respectively:
- `y`: vector of dependent variables
- `s_t`: vector of states
- `X`: matrix of exogenous variables (generated from standard normal distribution)

Function `add_lags(y::Vector{Float64}, p::Int64)` adds `p` lags to the matrix of dependent variables. The function returns a matrix of dependent variables with `p` lags.


## References

- Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. Econometrica: Journal of the Econometric Society, 357-384.

- Kim, Chang Jin (1994). Dynamic Linear Models with Markov-Switching. Journal of
Econometrics 60, 1-22.

- Filardo, Andrew J. (1994). Business cycle phases and their transitional dynamics. Journal of Business & Economic Statistics, 12(3), 299-308.

- Guidolin, Massimo & Pedio, Manuela (2018). Essentials of Time Series for Financial Applications. Academic Press.
