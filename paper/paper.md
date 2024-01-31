---
title: 'MarSwitching.jl: Julia package for Markov switching dynamic models'
tags:
  - Julia
  - Time series
  - Econometrics
  - Markov processes 
  - Nonlinear models
authors:
  - name: Mateusz Dadej
    orcid: 0000-0002-1791-7611
    corresponding: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
affiliations:
 - name: Phd. student, University of Brescia, Italy
   index: 1
 - name: Visiting Scholar, University of Mannheim, Germany
   index: 2
date: 30 February 2024
bibliography: paper.bib

---

# Summary

'MarSwitching.jl'([@bezanson2017julia]) is the first package in Julia programming language implementing Markov Switching Dynamic Models. It provides a set of tools for estimation, simulation and forecasting of Markov switching models. This class of models is the principal tool for modelling time series with regime changes. The time-variation of model parameters is governed by the limited memory Markov process. Because of non-trivial likelihood function and the amount of model parameters, Julia is a perfect language to implement this class of models due to its performance. 


# Statement of need

The Markov switching regression (also referred to as regime switching) was first introduced in the seminal work of [@hamilton89]. Since then, it has been extensively used in empirical research. Although the model was introduced as an application to economic data, the range of applications has expanded significantly since the first publication. These fields include finance [@buffington02], political science [@Brandt2014], hydrology [@wang23], epidemiology [@shiferaw21] and even bibliometrics [@delbianco20].

'MarSwitching.jl' is, at the moment, the only package dedicated to estimation of Markov switching models available in the 'Julia' programming language. At the same time, it is implemented purely in this language. 

# Background

Markov switching models are a class of regression models that allow for time variation of parameters in an otherwise linear model. More specifically, the current state is determined only by the state from the previous period, which is described in the transition matrix.

Consider a general model:
$$\mathbf{y}_t = \mathbf{X}_{t,i} \mathbf{\beta}_{S, i} + \mathbf{\epsilon}_t$$
$$\mathbf{\epsilon} \sim f(0,\mathbf{\Sigma}_s)$$
Where $\mathbf{y}_t$ is $N$ size vector of dependent variable indexed by time $t$. $\mathbf{X}_{t,i}$ is $N \times M$ matrix of exogenous regressors. $\mathbf{\beta}_{S, i}$ is $K$ size vector of parameters. $\mathbf{\epsilon}_t$ is $N$ size vector of errors. The errors are distributed according to some distribution $f(0,\mathbf{\Sigma}_s)$ with mean zero and covariance matrix $\mathbf{\Sigma}_s$. The state $S$ is a latent (unobservable) variable that can take values from $1$ to $K$. Parameters indexed by $S$ are different for each state.

The state $S_t$ is governed by the Markov process. The probability of transition from state $i$ to state $j$ is given by the $K \times K$ left-stochastic transition matrix $\mathbf{P}$:

\begin{equation*}
  \mathbf{P} = P(S_t = i | S_{t-1} = j) = 
    \begin{pmatrix}
    p_{1,1} & p_{1,2} & \cdots & p_{1,k} \\
    p_{2,1} & p_{2,2} & \cdots & p_{2,k} \\
    \vdots  & \vdots  & \ddots & \vdots  \\
    p_{k,1} & p_{k,2} & \cdots & p_{k,k} 
    \end{pmatrix}
\end{equation*}

With standard constraints: $0 < p_{i,j} < 1, \forall j,i \in \{1,\dots, K\}$ and $\sum_{i}^{K} p_{i,j} \forall j \in \{1, \dots, K\}$.

In a standard model, the transition matrix is assumed to be constant over time. However, it is possible to allow for time variation of the transition matrix itself, as described in [@filardo94] (and as implemented in the package). In this case, each of the transition probabilities is modeled as a function of the exogenous variables $\mathbf{Z}_{t}$:

\begin{equation*}
p_{i,j,t} = \dfrac{\exp(\delta_{i,j}'\mathbf{Z}_t)}{\textstyle \sum_{j=1} \exp(\delta_{i,j}'\mathbf{Z}_t)} 
\end{equation*}

Where $\delta_{i,j}$ is a vector of coefficients. The exponentiation and sum division of the coefficients ensure that the probabilities are non-negative and sum to one. For this model, the expected duration is time-varying as well.

# Main features

## Model estimation

There is a single function to estimate the model. With following simplified syntax

```
MSModel(y::VecOrMat{V},
        k::Int64,
        ;intercept::String = "switching",
        exog_vars::VecOrMat{V},
        exog_switching_vars::VecOrMat{V},
        switching_var::Bool = true,
        exog_tvtp::VecOrMat{V},
        random_search::Int64 = 0,
        random_search_em::Int64) where V <: AbstractFloat
```

With the mandatory arguments being:

- `y::VecOrMat{V}` - Data of dependent variable.
- `k::Int64` - Number of states. Needs to be at least 2. Since a model with `k=1` is equivalent to a standard linear regression model.

- `Intercept::String` - A string defining the intercept term. Possible values are `"switching"` (default), `"non-switching"` and `"no"` (for model without intercept)
- `exog_vars::VecOrMat{V}` - A vector or matrix of non-switching exogenous variables.
- `exog_switching_vars::VecOrMat{V}` - A vector or matrix of switching exogenous variables.
- `switching_var::Bool` - A boolean value indicating whether the variance of the error term is switching. Default is `true`.
- `exog_tvtp::VecOrMat{V}` - A vector or matrix of exogenous variables for time-varying transition probabilities. To define the intercept in the equation of transition probabilities, the user should include a column of ones in the matrix. On default the model assumes that the transition probabilities are constant over time.
- `random_search::Int64` - Number of random searches of the optimization (excluding the initial one). The optimization is ran each time with random changes to the initial values of the parameters. The best result is then chosen based on the log-likelihood value. Default is `0`, which means no random searches are performed.
- `random_search_em::Int64` - Number of random searches of the expectation-maximization algorithm. Default is `0`, which means no random searches are performed.

## Model summary


There are several functions for printing statistics of the estimated model. `state_coeftable(model::MSM, state::Int64; digits::Int64=3)` shows model coefficient's statistics for a given state and expected duration of the state. The standard errors are calculated using the inverse Hessian method [@penrose55]. For a standard model with a constant transition matrix, the function `transition_mat(model::MSM; digits::Int64=2)` prints a formatted matrix of estimated transition probabilities. For models with time-varying transition probabilities, the coefficients can be inspected with `coeftable_tvtp(model::MSM; digits::Int64=3)`.The functions above are building blocks of the main summary function `summary_mars(model::MSM)`, which prints all the relevant information about the model for each of the states from previous functions. Additionally, it shows basic information about the model and fitness statistics.

## Model simulation

It is possible to simulate the data either from an estimated model or for a specified parameter set. Thanks to `Julia`'s multiple dispatch, the function `generate_msm` may be called in two ways. Either with the model object as an argument:

`generate_msm(model::MSM, T::Int64=model.T)`

Or for the given parameters of the model:

`generate_msm(\mu::Vector{Float64}, 
              \sigma::Vector{Float64}, 
              P::Matrix{Float64}, 
              T::Int64; 
              \beta::Vector{Float64}, 
              \beta_ns::Vector{Float64}, 
              \delta::Vector{Float64}, tvtp_intercept::Bool)`

The function returns an object of type `tuple` containing the dependent variable `y::VectorFloat64`,
regimes time series `s_t::VectorFloat64`, and the design matrix `X::MatrixFloat64` ($X \sim N(0,1)$).

# Citations

# Acknowledgements

This open-source research software project received no financial support.

# References