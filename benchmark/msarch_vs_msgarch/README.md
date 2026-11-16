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
not a different optimum. MSGARCH starts each regime's variance at its
unconditional level (where `MSARCHModel` backcasts with the mean squared
residual), uses `y_1` only inside the variance recursion, and starts the
filter at `t = 2`, so the first observation's density is not in its
log-likelihood. Evaluating `MarSwitching.loglik` at MSGARCH's exact
estimates gives 4277.10, the same as `MSARCHModel`'s own optimum (4277.10).
Recomputed with MSGARCH's convention, MSGARCH's estimates give 4274.3433,
its reported value, and `MSARCHModel`'s give 4274.3443. So both packages
found the same optimum.

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
`random_search_em = 8, random_search = 8`: 3.5 s once compiled, or 2.0 s
with 6 threads (`julia -t 6`); the first call of a Julia session takes
about 15 s, because it includes JIT compilation. With
`algorithm = :LN_SBPLX`, which reaches the same estimates, that first call
took 93 s. MSGARCH's `FitML` used its defaults (a few seconds).

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
printed digit, so the two packages compute the same model.

The remaining difference is where each package's optimum lies, and each is
best on its own convention: under MarSwitching's, its point is 0.18 nats
above MSGARCH's (4312.78 vs 4312.60); under MSGARCH's, MSGARCH's point is
0.98 nats above MarSwitching's (4309.22 vs 4308.23). The burst regime's
persistence is weakly identified (a flat ridge), and the convention decides
where along it the maximum falls. MSGARCH's unconditional-variance start,
`omega / (1 - alpha - beta)`, grows without bound as the persistence
approaches 1, which plausibly keeps its estimate at 0.958.

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
random_search = 2`: 15 s once compiled (28 s for the first call of a Julia
session, including JIT compilation; with `algorithm = :LN_SBPLX`, which
reaches the same estimates, that first call took 335 s). MSGARCH's `FitML`
used its defaults (about 10 s). Standard errors in parentheses.

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

`speed_msarch.jl` / `speed_msgarch.R` time 20 repeated fits of the same
2-regime ARCH(1) model. Re-run 2026-11-15, MarSwitching.jl on `garch`
branch, MSGARCH 2.51, Windows 11, Intel Core i7-9850H.

| | mean | median | min | max | calls reaching the best optimum |
|---|---|---|---|---|---|
| MSGARCH `FitML` (R), defaults | 1.07s | 1.03s | 0.92s | 1.51s | 20/20 |
| `MSARCHModel`, defaults (`:LN_SBPLX`), random EM start | 2.35s | 1.06s | 0.97s | 12.10s | 18/20 |
| `MSARCHModel`, defaults (`:LN_SBPLX`), fixed start | 1.01s | 0.96s | 0.93s | 1.48s | 20/20 |
| `MSARCHModel`, `:LD_LBFGS`, random EM start | 0.62s | 0.57s | 0.47s | 1.10s | 15/20 |
| `MSARCHModel`, `:LD_LBFGS`, fixed start | 0.58s | 0.57s | 0.55s | 0.62s | 20/20 |

"Best optimum" is log-likelihood 4277.1046 in MarSwitching's convention,
which is MSGARCH's 4274.3433 (see "On the log-likelihood gap" above).

- **The two packages now take about the same time**, although they do
  different amounts of work: a default `MSARCHModel` call runs 5 NLopt
  optimizations (one from the EM start plus 4 structured restarts over
  total ARCH persistence), while `FitML` runs one. The default's median is
  1.06s against MSGARCH's 1.03s, and `:LD_LBFGS` from a fixed start is
  about 1.8x faster than `FitML`.
- **The random EM start matters more than the optimizer.** By default the
  EM starting point is drawn from the global RNG on every call (a "fixed
  start" here re-seeds the RNG before each call). With a random start, 2
  of 20 default calls and 5 of 20 `:LD_LBFGS` calls ended in a worse local
  optimum (4225.26, about 52 nats lower), and the slowest default call took
  12.1s. MSGARCH starts from the same deterministic point every time. Pass
  `random_search_em` (several EM starts, keeping the best) or
  `random_search` when it matters.
- **Earlier measurements.** In August 2026 the defaults averaged 14.3s
  and `:LD_LBFGS` 4.6s on the same data. Since then the Hamilton filter
  shared by all models was rewritten (the ARCH likelihood alone became
  about 2.6x faster) and other code changed, so these old numbers are a
  reference point, not a clean before/after of one change.

Julia additionally pays a one-time JIT compilation cost on the first call
of a session (15.4s here) that R does not have; irrelevant once warmed up
(e.g. inside a longer-running process fitting many models), but worth
knowing for single-shot script runs.
