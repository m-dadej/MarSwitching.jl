```@meta
Description = "How MarSwitching.jl compares to MSwM, MSGARCH and dynr in R, statsmodels in Python, MS_Regress in MATLAB, and to EViews, Stata and HiddenMarkovModels.jl — features, speed and when to use each."
```

# Comparison with other Markov switching software

Markov switching models are available in most statistical ecosystems, and which
implementation is the right one depends on the model you actually need. This page
sets out what MarSwitching.jl does and does not cover relative to the established
alternatives, so that the choice can be made on features rather than on familiarity.

## The landscape

| Software | Language | Scope |
|:---|:---|:---|
| [`MSwM`](https://cran.r-project.org/package=MSwM) | R | Univariate autoregressive Markov switching models, estimated by EM. Gaussian, Poisson, binomial and gamma responses. |
| [`MSGARCH`](https://cran.r-project.org/package=MSGARCH) | R | Markov switching GARCH-type models specifically, with MLE and Bayesian/MCMC estimation and risk measures (VaR, expected shortfall). |
| [`dynr`](https://cran.r-project.org/package=dynr) | R | Regime-switching nonlinear dynamic systems and state space models. |
| `statsmodels` (`MarkovRegression`, `MarkovAutoregression`) | Python | Markov switching regression and autoregression with constant transition probabilities. |
| [`MS_Regress`](https://github.com/msperlin/MS_Regress-Matlab) | MATLAB | Markov switching regression, including some GARCH specifications. |
| EViews, Stata (`mswitch`), SAS | Commercial | Markov switching regression and autoregression, with GUI workflows. |
| [`HiddenMarkovModels.jl`](https://github.com/gdalle/HiddenMarkovModels.jl) | Julia | General-purpose HMM inference — Baum-Welch, Viterbi, forward-backward — over arbitrary observation distributions. |
| **MarSwitching.jl** | Julia | Markov switching regression with `k` regimes, per-parameter switching control, time-varying transition probabilities and MS-ARCH. |

## Where MarSwitching.jl is distinctive

- **Time-varying transition probabilities (TVTP)** à la Filardo (1994), where exogenous
  variables drive the transition matrix itself. This is not available in `MSwM` or in
  `statsmodels`, and it is the feature most often missing when researchers move from a
  constant-transition model to an applied one. See the
  [stock market example](examples/example_spx.md).
- **Per-parameter switching control.** The intercept, the variance, individual covariates
  and the shape parameter of the error distribution can each be declared switching or
  non-switching independently, rather than the whole equation switching together.
- **Non-Gaussian errors** — Student's ``t`` and the Generalized Error Distribution, with a
  regime-switching shape parameter.
- **Markov switching ARCH** (`MSARCHModel`), with regime-specific ARCH(``q``) coefficients
  following Haas, Mittnik & Paolella (2004).
- **Speed.** See below.

## Performance

The benchmark below was run on artificially generated data with 400 observations, from a
model with 3 regimes, 1 switching and 1 non-switching exogenous variable. The table reports
the mean absolute error of the estimated parameters with respect to the true parameters used
by `generate_msm()`, so that speed is compared at equal accuracy.

|                |MarSwitching.jl| statsmodels  | MSwM     | MS_Regress     |
|:---------------|-------------:|--------------:|---------:|---------------:|
| implementation | Julia        | Python/Cython | R        | MATLAB/MEX/C++ |
| error:         |              |               |          |                |
| mu             | 0.0363       | 0.0363        | 0.036    | 0.0367         |
| beta_s         | 0.0237       | 0.0237        | 0.0245   | 0.0241         |
| beta_ns        | 0.0150       | 0.01508       | 0.0211   | 0.0157         |
| sigma          | 0.0083       | 0.0083        | 0.0108   | 0.0084         |
| p              | 0.0138       | 0.0138        | 0.0157   | 0.0139         |
|                |              |               |          |                |
| runtime (s)    | 0.471        | 3.162         | 3.867    | 19.959         |
| relative       | 1            | 6.713         | 8.21     |    42.376      |

Every implementation reached virtually the same estimation error, while MarSwitching.jl was
6.7 times faster than `statsmodels`, 8.2 times faster than `MSwM` and 42 times faster than
`MS_Regress` — although the MATLAB package also computes standard errors within the same
call.

Software versions: MarSwitching.jl v0.2.2, statsmodels v0.14.1, MSwM v1.5, MS_Regress v1.11,
on Julia v1.10.1, Python v3.12.2, R v4.2.1 and MATLAB vR2024a. Calculations were run on
Windows 11 x64, Intel(R) Core(TM) i7-9850H @ 2.60GHz, 6 cores. The benchmark code is in the
`benchmark` folder of the repository.

## When to use something else

MarSwitching.jl is not the right tool for every regime switching problem:

- **You need MS-GARCH rather than MS-ARCH.** `MSGARCH` in R is the mature choice, and it
  additionally offers Bayesian estimation and built-in risk measures. MS-GARCH is on the
  MarSwitching.jl roadmap but is not implemented yet.
- **You need a Markov switching VAR.** Not yet implemented here; this is also on the roadmap.
- **You need autoregressive dynamics with lagged states**, i.e.
  ``y_t = \mu_{S_t} + \phi(y_{t-1} - \mu_{S_{t-1}})``. `MSwM` and the `statsmodels`
  `MarkovAutoregression` class cover this; MarSwitching.jl currently does not.
- **You want general-purpose hidden Markov model inference** over arbitrary emission
  distributions, rather than a switching regression. `HiddenMarkovModels.jl` is the better
  fit within Julia.
- **You need Bayesian estimation.** MarSwitching.jl estimates by maximum likelihood only.

## References

Filardo, A. J. (1994). Business-cycle phases and their transitional dynamics.
*Journal of Business & Economic Statistics*, 12(3), 299-308.

Haas, M., Mittnik, S., & Paolella, M. S. (2004). A new approach to Markov-switching GARCH
models. *Journal of Financial Econometrics*, 2(4), 493-530.

Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series
and the business cycle. *Econometrica*, 57(2), 357-384.
