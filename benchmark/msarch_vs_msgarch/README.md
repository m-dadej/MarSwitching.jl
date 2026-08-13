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

**On the log-likelihood gap (~2.8 nats):** `MSARCHModel` found the
slightly *better* optimum here, most likely because it was run with
aggressive multi-start while MSGARCH's default `FitML` is a single run.
MS-ARCH likelihoods are known to be multimodal (see the note in
`MSModel`'s docstring), and are especially flat near `alpha = 0` — which
is exactly the calm regime's estimate here, so both packages landing on
slightly different points along a near-flat ridge is expected, not a
correctness concern. Re-running MSGARCH with more restarts (or
`MSARCHModel` with `random_search = 0`) narrows the gap further.

**Transition-matrix convention:** MSGARCH reports its 2x2 matrix
row-stochastic (`P_1_1` = P(stay in 1)); `MSARCHModel`'s `model.P` is
left-stochastic (`P[i,j]` = P(state i at t | state j at t-1), columns sum
to 1 — see the `MSM` docstring). The two conventions agree on the
diagonal (persistence probabilities), which is what's compared above.

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
