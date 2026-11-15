# MSARCHModel vs. MSGARCH: a cross-implementation check

Fits the same 2-regime, ARCH(1), zero-mean model to the same data with two
independent implementations — `MSARCHModel()` from this package (Julia) and
the `sARCH` model in R's [MSGARCH](https://github.com/keblu/MSGARCH) package
([Ardia, Bluteau, Boudt, Catania & Trottier, 2019, *JSS*](https://www.jstatsoft.org/article/view/v091i04))
— and compares the estimated parameters. Close agreement between two
unrelated codebases (different language, different optimizer, different
likelihood implementation) is evidence that `MSARCHModel`'s estimation is
correct, not just internally self-consistent.

The model (Haas, Mittnik & Paolella, 2004; nested from Hamilton & Susmel,
1994's original MS-ARCH):

```
y_t = e_t,   e_t = sqrt(h_{t,S_t}) * z_t,   z_t ~ N(0,1)
h_{t,s} = omega_s + alpha_s * y_{t-1}^2
```

with `S_t` a 2-state Markov chain. Zero mean (`intercept = "no"` on the
Julia side) because MSGARCH's variance-only models have no mean/location
submodel, so this is the only specification directly comparable between
the two packages without introducing an extra degree of freedom on one
side.

## Data

`spx_weekly_returns.csv` — weekly S&P 500 log returns, 1991-04-30 to
2023-11-06 (1,709 observations). Extracted from this package's own
`docs/src/man/examples/my_assets/df_spx.csv` (used in the TVTP example in
the docs), keeping only the date and `spx` columns needed here.

## Running it

```bash
# Julia side (uses this package's own Project.toml/Manifest.toml)
julia benchmark/msarch_vs_msgarch/fit_msarch.jl

# R side (requires MSGARCH: install.packages("MSGARCH"))
Rscript benchmark/msarch_vs_msgarch/fit_msgarch.R
```

Both scripts resolve paths relative to their own location, so they can be
run from any working directory.

## Reference results

Obtained 2026-08-12, MarSwitching.jl on `arch_model` branch, MSGARCH 2.51.
`MSARCHModel` was run with `random_search_em = 8, random_search = 8`
(multi-start); MSGARCH's `FitML` used its defaults (single optimization
run from its own internal starting values).

| Parameter | MSGARCH (R) | MSARCHModel (Julia) |
|---|---|---|
| omega, regime 1 (calm) | 1.8807e-04 | 1.8820e-04 |
| alpha, regime 1 | 2.640e-05 (~0) | 1.7e-16 (~0) |
| omega, regime 2 (volatile) | 8.3112e-04 | 8.3128e-04 |
| alpha, regime 2 | 0.15377 | 0.15379 |
| P(stay in regime 1) | 0.9802 | 0.9803 |
| P(stay in regime 2) | 0.9704 | 0.9704 |
| expected duration, regime 1 (weeks) | 50.5 | 50.8 |
| expected duration, regime 2 (weeks) | 33.7 | 33.8 |
| log-likelihood | 4274.34 | 4277.10 |

Core parameters (omega, alpha, transition probabilities) agree to
3-4 significant figures.

**On the log-likelihood gap (~2.8 nats):** this is a reporting convention,
not a different optimum. MSGARCH leaves the first observation's density
out of the log-likelihood it reports (and starts each regime's variance at
its unconditional level, where `MSARCHModel` backcasts with the mean
squared residual). Evaluating `MarSwitching.loglik` at MSGARCH's exact
estimates gives 4277.10, the same as `MSARCHModel`'s own optimum, and
4274.39 once the first observation is dropped, against MSGARCH's 4274.34.
So both packages found the same optimum.

## MSGARCHModel vs. MSGARCH `sGARCH`

`fit_msgarch11.jl` and `fit_msgarch11.R` repeat the comparison for the
2-regime, zero-mean GARCH(1,1) model on the same data:

```
y_t = e_t,   e_t = sqrt(h_{t,S_t}) * z_t,   z_t ~ N(0,1)
h_{t,s} = omega_s + alpha_s * y_{t-1}^2 + gamma_s * h_{t-1,s}
```

MSGARCH calls the lagged-variance coefficient `beta`; MarSwitching.jl
calls it `gamma` because `β` already holds regression coefficients.

```bash
julia benchmark/msarch_vs_msgarch/fit_msgarch11.jl
Rscript benchmark/msarch_vs_msgarch/fit_msgarch11.R
```

Obtained 2026-11-15, MarSwitching.jl on `garch` branch, MSGARCH 2.51.
`MSGARCHModel` was run with its default optimizer (`:LD_SLSQP`) and
`random_search_em = 8, random_search = 8` (16 s; 93 s with
`algorithm = :LN_SBPLX`, which reaches the same estimates); MSGARCH's
`FitML` used its defaults (a few seconds).

| Parameter | MSGARCH (R) | MSGARCHModel (Julia) |
|---|---|---|
| omega, regime 1 (calm, persistent) | 6.097e-06 | 5.343e-06 |
| alpha, regime 1 | 0.0658 | 0.0649 |
| beta / gamma, regime 1 | 0.8980 | 0.8997 |
| omega, regime 2 (short volatility bursts) | 6.807e-04 | 5.773e-04 |
| alpha, regime 2 | 0.4334 | 0.4404 |
| beta / gamma, regime 2 | 0.5244 | 0.5596 |
| P(stay in regime 1) | 0.9504 | 0.9448 |
| P(stay in regime 2) | 0.3581 | 0.3463 |
| expected duration, regime 1 (weeks) | 20.2 | 18.1 |
| expected duration, regime 2 (weeks) | 1.56 | 1.53 |
| log-likelihood (as reported) | 4309.22 | 4312.78 |

For the persistent regime, alpha and gamma agree to 2-3 significant
figures; omega, which is tiny and trades off against the persistence,
agrees to within ~15%. The burst regime is less precisely determined. It lives 1.5 weeks on average, so only
about 7% of the sample informs it. `MSGARCHModel` puts it at the
stationarity bound (`alpha + gamma` = 0.99994, the cap that keeps the
optimizer's steps finite; `summary_msm` therefore reports no standard
errors for that regime's `alpha` and `gamma`), while MSGARCH stops at 0.958.

**On the log-likelihood gap (~3.6 nats):** as for ARCH, ~3.4 nats are the
reporting convention. `MarSwitching.loglik` at MSGARCH's exact estimates
gives 4312.60. MSGARCH instead starts each regime's variance at its
unconditional level, uses `y_1` only inside the variance recursion, and
starts the filter at `t = 2` from the ergodic probabilities; recomputed that
way, the same parameters give 4309.2195, MSGARCH's reported value to every
printed digit. The remaining 0.18 nats are a genuinely better point, found by
moving along the flat ridge towards the stationarity bound of regime 2.

**Transition-matrix convention:** MSGARCH reports its 2x2 matrix
row-stochastic (`P_1_1` = P(stay in 1)); `MSARCHModel`'s `model.P` is
left-stochastic (`P[i,j]` = P(state i at t | state j at t-1), columns sum
to 1 — see the `MSM` docstring). The two conventions agree on the
diagonal (persistence probabilities), which is what's compared above.

## MS-GARCH(1,1) on simulated data with known parameters

The S&P 500 comparison above is hard to judge: on the weekly-return scale
`omega` is tiny (it prints as 0.0 in `summary_msm`), and the burst regime
sits at the stationarity bound. Here both packages fit simulated data
instead, so their estimates can be compared with each other *and* with the
true parameters, all of which are clearly non-zero and interior.

**Data.** `sim_msgarch11_data.csv` holds T = 10,000 draws from a 2-regime,
zero-mean MS-GARCH(1,1) (plus the true regime path), simulated by
`sim_msgarch11_generate.jl` with `generate_msm()`. The true parameters are
in `sim_msgarch11_params.csv`, named as in MSGARCH. Regime 1 is calm
(unconditional variance 1.0), regime 2 turbulent (6.67); each lasts 100
periods on average. The design was picked so that the standard errors are
about 3-11% of the true `alpha` and `beta`, and 15-17% of `omega` (GARCH
intercepts are imprecise even at this sample size).

```bash
julia   benchmark/msarch_vs_msgarch/sim_msgarch11_fit.jl
Rscript benchmark/msarch_vs_msgarch/sim_msgarch11_fit.R
Rscript benchmark/msarch_vs_msgarch/sim_msgarch11_mc.R    # optional, ~7 min
```

Both fit scripts sort the regimes by unconditional variance (the regime
labels are arbitrary) and print the same table. Obtained 2026-11-15,
MarSwitching.jl on `garch` branch, MSGARCH 2.51. `MSGARCHModel` was run with
its default optimizer (`:LD_SLSQP`) and `random_search_em = 2,
random_search = 2` (28 s; 335 s with `algorithm = :LN_SBPLX`, which reaches
the same estimates); MSGARCH's `FitML` used its defaults (about 10 s).
Standard errors in parentheses.

| Parameter | True | MSGARCH (R) | MSGARCHModel (Julia) |
|---|---|---|---|
| omega, regime 1 (`alpha0_1`) | 0.10 | 0.0927 (0.0143) | 0.0931 (0.0143) |
| alpha, regime 1 (`alpha1_1`) | 0.15 | 0.1313 (0.0257) | 0.1311 (0.0198) |
| beta / gamma, regime 1 (`beta_1`) | 0.75 | 0.7659 (0.0169) | 0.7650 (0.0229) |
| omega, regime 2 (`alpha0_2`) | 1.00 | 1.0811 (0.1213) | 1.0782 (0.1210) |
| alpha, regime 2 (`alpha1_2`) | 0.25 | 0.2658 (0.0349) | 0.2656 (0.0189) |
| beta / gamma, regime 2 (`beta_2`) | 0.60 | 0.5671 (0.0217) | 0.5676 (0.0288) |
| P(stay in regime 1) (`P_1_1`) | 0.99 | 0.9865 (0.0017) | 0.9864 (0.0025) |
| P(regime 2 to 1) (`P_2_1`) | 0.01 | 0.0093 (0.0025) | 0.0093 (0.0017) |
| log-likelihood, MSGARCH convention | -19479.30 at the truth | -19475.9795 | -19475.9853 |
| regimes correctly classified (smoothed) | | 90.7% | 90.7% |

**Estimates.** The two packages agree to 2-3 significant figures on every
parameter (the largest relative gap is 0.4%, for regime 1's `omega`), and
both are within about 2 standard errors of the truth. Under MSGARCH's
likelihood convention (see the S&P section) the two optima differ by 0.006
nats: each package maximises its own convention, so each is marginally
ahead on its own objective. They are the same optimum, and both are 3.3
nats above the likelihood at the true parameters, as expected of a
maximum. Both classify 90.7% of periods into the right regime.

**Standard errors.** These differ by up to 2x between the packages, so
`sim_msgarch11_mc.R` checks them against the actual sampling spread of the
estimator: 40 fresh samples from the same process, each fitted with
MSGARCH. Since both packages reach the same optimum, that spread is the
benchmark for both.

| Parameter | Monte Carlo SD | MSGARCH SE (median over 40 fits) | MSGARCHModel SE (the one sample above) |
|---|---|---|---|
| `alpha0_1` | 0.0106 | 0.0146 (+38%) | 0.0143 (+35%) |
| `alpha1_1` | 0.0177 | 0.0268 (+51%) | 0.0198 (+12%) |
| `beta_1`   | 0.0187 | 0.0172 (-8%)  | 0.0229 (+22%) |
| `alpha0_2` | 0.1089 | 0.1400 (+29%) | 0.1210 (+11%) |
| `alpha1_2` | 0.0182 | 0.0376 (+107%) | 0.0189 (+4%) |
| `beta_2`   | 0.0233 | 0.0233 (0%)   | 0.0288 (+24%) |
| `P_1_1`    | 0.0019 | 0.0020 (+3%)  | 0.0025 (+32%) |
| `P_2_1`    | 0.0019 | 0.0021 (+11%) | 0.0017 (-10%) |

MSGARCH overstates the `alpha` standard errors by 1.5-2x. MSGARCHModel's
are within about +-35% of the Monte Carlo SD for every parameter, but they
come from a single sample, whose own SEs vary from sample to sample, and a
40-sample SD is itself uncertain by about 11%. So read them as "the right
size", not as more precise than that.

## Speed comparison

`speed_msarch.jl` / `speed_msgarch.R` time repeated single-optimization
fits of the same model (`MSARCHModel`'s defaults: one EM initialization +
one NLopt run, to match `FitML()`'s default of one run from one starting
point).

| | mean | median | min | max |
|---|---|---|---|---|
| MSGARCH `FitML` (R) | 1.07s | 1.02s | 0.83s | 1.85s |
| `MSARCHModel`, defaults (`:LN_SBPLX`, random EM start) | 14.31s | 12.07s | 3.75s | 50.99s |
| `MSARCHModel`, `:LD_LBFGS` (random EM start) | 4.56s | 4.53s | 1.51s | 9.46s |
| `MSARCHModel`, `:LD_LBFGS`, fixed start | 0.92s | 0.88s | — | — |

At the literal defaults, MSGARCH is ~13x faster — but that gap is not a
Julia-vs-R difference, it's an optimizer-choice difference, and it
collapses once the comparison controls for it:

- `MSARCHModel` defaults to NLopt's derivative-free `:LN_SBPLX` with a
  tightened `xtol_rel = 1e-6` for ARCH models (vs `1e-4` for the
  constant-variance model) — a deliberate robustness choice, since MS
  likelihoods are prone to local optima (see `MSModel`'s docstring). A
  gradient-based algorithm generally needs far fewer function evaluations
  to reach the same tolerance.
- The default EM-based starting point is redrawn (unseeded) on every
  call, so back-to-back fits aren't repeated runs from the same point —
  this alone explains most of the variance (3.75s-51.0s spread with
  `:LN_SBPLX` vs. 5.6s mean from a *fixed* start).
- Switching to `algorithm = :LD_LBFGS` (gradient-based, built into
  `MSModel`/`MSARCHModel`, gradients via `FiniteDiff`) alone cuts the mean
  from 14.3s to 4.6s. Combined with a fixed starting point (removing the
  EM-randomness noise), `:LD_LBFGS` fits in ~0.9s — matching or slightly
  *beating* MSGARCH's default, landing on the identical log-likelihood
  (4277.104559336357 vs 4277.104559336371, i.e. the same optimum to
  10 significant figures).

Julia additionally pays a one-time JIT compilation cost on the first call
of a session (~48s for `:LN_SBPLX`, ~24s for `:LD_LBFGS` here) that R does
not have an equivalent of; irrelevant once warmed up (e.g. inside a
longer-running process fitting many models), but worth knowing for
single-shot script runs.

**Practical takeaway:** the package's default (`:LN_SBPLX` + randomized EM
start) trades speed for robustness against the multimodality of MS
likelihoods, which is a reasonable default especially when paired with
`random_search`/`random_search_em` multi-start. For a single well-chosen
starting point where robustness is less of a concern, `algorithm =
:LD_LBFGS` is on par with MSGARCH's default speed.
