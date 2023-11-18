# MarSwitching.jl: Markov Switching dynamic models in Julia

[![Build Status](https://github.com/m-dadej/MARS.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/m-dadej/MARS.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Build status](https://ci.appveyor.com/api/projects/status/ff0w59c7vlm0600t?svg=true)](https://ci.appveyor.com/project/m-dadej/marswitching-jl)
[![codecov](https://codecov.io/gh/m-dadej/MarSwitching.jl/graph/badge.svg?token=AANR7304QU)](https://codecov.io/gh/m-dadej/MarSwitching.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MarSwitching.jl is a package for estimating Markov switching dynamic models (also called regime switching) for Julia. This is a class of models with time-varying coefficients depending on an unobservable state/regime that follows Markov process. The package provides tools for estimation, inference and simulation of the models. 

Author: Mateusz Dadej, m.dadej at unibs.it

!!! info "Star it on GitHub!"
    If you have found this package useful, please consider starring it on [GitHub](https://github.com/m-dadej/MarSwitching.jl).
    ```@raw html
    <script async defer src="https://buttons.github.io/buttons.js"></script>

    <a class="github-button" 
    href="https://github.com/m-dadej/MarSwitching.jl" 
    data-icon="octicon-star" 
    data-size="large" 
    data-show-count="true" 
    aria-label="Star alan-turing-institute/MLJ.jl on GitHub">
    Star</a>
    ```

## Installation
MarSwitching is in general registry. To install simply enter `]` in the Julia's REPL and use following command:

```julia
pkg> add MarSwitching
```
Assuming that you already have at least Julia 1.6 (stable version) installed.

## Functionality 

- Currently available:
    - Markov switching model with $k$ regimes and combinations of switching/non-switching:
        - intercept
        - variance
        - exogenous variables
    - model with time-varying transition probabilities (TVTP) (à la Filardo 1994) 
    - Filtered probabilities
    - Smoothed probabilities (Kim, 1994)
    - Summary statistics of coefficients
    - instantaneous and one step ahead `predict()`
    - Expected regime duration
    - Simulation of data both from estimated model and from given parameters
    - variable and number of states selection (with random and grid search)
- Planned functionality:
    - other error distributions (student-t, GED, etc.)
    - Markov Switching GARCH model
    - Markov Switching VAR model
    - Markov Switching model with lagged states. E.g. $y_t = \mu_{S_t} + \phi(y_{t-1} - \mu_{S_{t-1}})$

Future development is closely related to the package's popularity.

## Performance comparison    

`MarSwitching.jl` is the fastest open source implementation of the model. The benchmark was done on artificially generated data with 400 observations, from the model with 3 regimes, 1 switching and 1 non switching exogenous variable. Table below shows mean absolute error of estimated parameters with respect to the actual parameters from `generate_msm()` function.

|                |MarSwitching.jl| statsmodels  | MSwM     | MS_Regress     |
|:---------------|-------------:|--------------:|---------:|---------------:|
| implementation | Julia        | Python/Cython | R        | Matlab/MEX/C++ |
| error:         |              |               |          |                |
| mu             | 0,0363       | 0,0363        | 0,036    | 0,0367         |
| beta_s         | 0,0237       | 0,0237        | 0,0245   | 0,0241         |
| beta_ns        | 0,0150       | 0,01508       | 0,0211   | 0,0157         |
| sigma          | 0,0083       | 0,0083        | 0,0108   | 0,0084         |
| p              | 0,0138       | 0,0138        | 0,0157   | 0,0139         |
|                |              |               |          |                |
| runtime (s)    | 0,471        | 3,162         | 3,867    | 19,959         |
| relative       | 1            | 6,713         | 8,21     |    42,376      |


`MarSwitching.jl` is 6,7 times faster than `statsmodels` implementation in `Python`/`Cython`, 8,2 times faster than `MSwM` in `R` and 42 times faster than `MS_Regress` in `MATLAB`/`MEX`, although MATLAB package is also calculating standard errors during function call. Every implementation had virtually the same error of estimated parameters.

Code of the benchmarks can be found in `benchmark` folder.

## Markov regime switching model in a nutshell

The Markov switching models are a class of models that allow for the parameters to change over time, depending on the unobservable state like economic recession, high volatility on financial markets or epidemiologic outbreak. The state follows Markov process with a given probability transition matrix for each of $k$ states:

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

Satisfying standard Markovian properties. The general model is defined as follows:

```math
\begin{align*}
\mathbf{y}_t &= \mathbf{\mu}_S + \mathbf{\beta}_{S}' \mathbf{X}_t + \mathbf{\gamma}'\mathbf{Z}_t + \mathbf{\epsilon}_t, & \mathbf{\epsilon} \sim f(0,\mathcal{\Sigma}_s)\\
\end{align*}
```

Where $\mathbf{y}_t$ is vector of dependent variables, $\mathbf{\mu}_s$ and $\mathbf{\beta}_s$ are model parameters dependent on the state $S_t$, $\mathbf{\gamma}$ is a vector of parameters for exogenous variables. The error is distributed according to some distribution $f$ with state dependent covariance matrix $\mathcal{\Sigma}_s$. 

Because of the unobserved nature of the state, the model is estimated by maximum likelihood. The likelihood function is calculated using the method described in Hamilton, 1989.

The package also provide time-varying transition probabilities (TVTP) (Filardo, 1994) which allows for the transition matrix to change over time. Each transition probability has a following form:

```math
p_{i,j,t} = \dfrac{exp(\delta_{i,j}'\mathbf{Z}_t)}{\textstyle \sum_{j=1} exp(\delta_{i,j}'\mathbf{Z}_t)}
```

