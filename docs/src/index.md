```@meta
Description = "MarSwitching.jl estimates Markov switching (regime switching) dynamic regression models in Julia: k regimes, time-varying transition probabilities, MS-ARCH, filtered and smoothed probabilities, by maximum likelihood."
```

# MarSwitching.jl: Markov Switching dynamic models in Julia

[![docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://m-dadej.github.io/MarSwitching.jl/stable)
[![Build Status](https://github.com/m-dadej/MarSwitching.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/m-dadej/MarSwitching.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/m-dadej/MarSwitching.jl/graph/badge.svg?token=AANR7304QU)](https://codecov.io/gh/m-dadej/MarSwitching.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![status](https://joss.theoj.org/papers/f0b33a8a4b30b3d9f0184dec014eb388/status.svg)](https://joss.theoj.org/papers/f0b33a8a4b30b3d9f0184dec014eb388)

MarSwitching.jl is a Julia package for estimating **Markov switching dynamic models** (also called **regime switching** or hidden Markov regression models). This is a class of time series models whose coefficients change over time with an unobservable state, or regime, that follows a Markov process. Such models are widely used in econometrics and quantitative finance — for business cycle and recession dating, volatility regime detection and forecasting — as well as in political science, hydrology and epidemiology. The package provides tools for estimation, inference and simulation of the models.

**Author**: [Mateusz Dadej](https://m-dadej.github.io/), mateuszdadej {at} gmail.com

!!! info "Star it on GitHub!"
    If you have found this package useful, please consider starring it on [GitHub](https://github.com/m-dadej/MarSwitching.jl).
    ```@raw html
    <script async defer src="https://buttons.github.io/buttons.js"></script>

    <a class="github-button" 
    href="https://github.com/m-dadej/MarSwitching.jl" 
    data-icon="octicon-star" 
    data-size="large" 
    data-show-count="true" 
    aria-label="Star m-dadej/MarSwitching.jl on GitHub">
    Star</a>
    ```

**citation**: I encourage to cite the [JOSS paper](https://joss.theoj.org/papers/10.21105/joss.06441) of the package when using it in your research. You can use the following BibTeX entry from the `CITATION.bib` file:

```bibtex
@article{Dadej2024, 
  doi = {10.21105/joss.06441}, 
  url = {https://doi.org/10.21105/joss.06441}, 
  year = {2024}, 
  publisher = {The Open Journal}, 
  volume = {9}, 
  number = {98}, 
  pages = {6441}, 
  author = {Mateusz Dadej}, 
  title = {MarSwitching.jl: A Julia package for Markov switching dynamic models}, 
  journal = {Journal of Open Source Software} 
}
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
        - shape of error distribution
    - Markov Switching ARCH model (`MSARCHModel()`), with regime-specific ARCH($q$) coefficients (Haas, Mittnik & Paolella, 2004)
    - Model with time-varying transition probabilities (TVTP) (à la Filardo 1994) 
    - Alternative error distributions (Normal, Student's $t$-distribution and Generalized Error Distribution) with regime-switching shape parameter
    - Filtered probabilities
    - Smoothed probabilities (Kim, 1994)
    - Summary statistics of coefficients
    - Instantaneous and one step ahead `predict()`
    - Expected regime duration
    - Simulation of data both from estimated model and from given parameters
    - Variable and number of states selection (with random and grid search)
- Planned functionality:
    - Markov Switching GARCH model
    - Markov Switching VAR model
    - Markov Switching model with lagged states. E.g. $y_t = \mu_{S_t} + \phi(y_{t-1} - \mu_{S_{t-1}})$

Future development is closely related to the package's popularity.

## Performance    

`MarSwitching.jl` is the fastest open source implementation of the model — 6.7 times faster than `statsmodels` in Python, 8.2 times faster than `MSwM` in R and 42 times faster than `MS_Regress` in MATLAB, at virtually the same estimation error.

See [Comparison with other Markov switching software](man/comparison.md) for the full benchmark and for how the package compares with `MSwM`, `MSGARCH`, `statsmodels`, `HiddenMarkovModels.jl` and the commercial alternatives.

## Contributing

- PRs with fixed bugs or new methods are highly appreciated. Especially the ones described in the [Functionality](#Functionality) section.
- Open an issue if the PR changes current code substantially.
- See [CONTRIBUTING.md](https://github.com/m-dadej/MarSwitching.jl/blob/main/CONTRIBUTING.md), and if unsure, check the [ColPrac](https://github.com/SciML/ColPrac) guide on collaborative practices for Packages.


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

