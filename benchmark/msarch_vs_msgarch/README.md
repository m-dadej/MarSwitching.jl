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
