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

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

'MarSwitching.jl'([@bezanson2017julia]) is the first package in Julia programming language implementing Markov Switching Dynamic Models. It provides a set of tools for estimation, simulation and forecasting of Markov switching models. This class of models is the principal tool for modelling time series with regime changes. The time-variation of model parameters is governed by the limited memory Markov process. Because of non-trivial likelihood function and the amount of model parameters, Julia is a perfect language to implement this class of models due to its performance. 


# Statement of need

The Markov switching regression (also referred to as regime switching) was first introduced in the seminal work of [@hamilton89]. Since then, it has been extensively used in empirical research. Although the model was introduced as an application to economic data, the range of applications has expanded significantly since the first publication. These fields include finance [@buffington02], political science [@Brandt2014], hydrology [@wang23], epidemiology [@shiferaw21] and even bibliometrics [@delbianco20].

'MarSwitching.jl' is, at the moment, the only package dedicated to estimation of Markov switching models available in the 'Julia' programming language. At the same time, it is implemented purely in this language. 

# Background

Markov switching models are a class of regression models that allow for time variation of parameters in an otherwise linear model. More specifically, the current state is determined only by the state from the previous period, which is described in the transition matrix.



# Main features

## Model estimation

There is a single function to estimate the model. With following simplified syntax"

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

# Citations

# Acknowledgements

This open-source research software project received no financial support.

# References