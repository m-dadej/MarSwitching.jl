# Monte Carlo check of standard errors for the simulated MS-GARCH(1,1) design in
# sim_msgarch11_generate.jl: draws R fresh samples of T = 10,000 from the same process,
# fits each with MSGARCH, and compares the spread of the estimates across samples (the
# actual sampling standard deviation) with the standard errors MSGARCH reports.
#
# MSGARCHModel() reaches the same optimum as MSGARCH on the same data (see README.md), so
# the Monte Carlo standard deviation is also the benchmark for MarSwitching.jl's standard
# errors. Fitting MSGARCHModel() 40 times at T = 10,000 would take hours, hence R here.
#
# Requires the MSGARCH package: install.packages("MSGARCH"). Takes about 7 minutes.
#
# Run from anywhere, e.g.:  Rscript benchmark/msarch_vs_msgarch/sim_msgarch11_mc.R

if (!requireNamespace("MSGARCH", quietly = TRUE)) {
  stop("MSGARCH is not installed. Install it first with: install.packages(\"MSGARCH\")")
}
library(MSGARCH)

script_dir <- dirname(sub("--file=", "", commandArgs()[grep("--file=", commandArgs())]))
truth <- read.csv(file.path(script_dir, "sim_msgarch11_params.csv"))
p     <- setNames(truth$value, truth$name)

omega <- c(p[["alpha0_1"]], p[["alpha0_2"]])
alpha <- c(p[["alpha1_1"]], p[["alpha1_2"]])
beta  <- c(p[["beta_1"]],   p[["beta_2"]])
# row-stochastic: P[i, j] = P(S_t = j | S_{t-1} = i)
P <- matrix(c(p[["P_1_1"]], 1 - p[["P_1_1"]],
              p[["P_2_1"]], 1 - p[["P_2_1"]]), 2, byrow = TRUE)

# same process as generate_msm(): one realised y_t drives every regime's variance recursion
simulate_msgarch <- function(n) {
  s <- integer(n); s[1] <- 1L
  for (t in 2:n) s[t] <- if (runif(1) < P[s[t - 1], 1]) 1L else 2L
  h <- omega / (1 - alpha - beta)
  y <- numeric(n)
  for (t in 1:n) {
    if (t > 1) h <- omega + alpha * y[t - 1]^2 + beta * h
    y[t] <- sqrt(h[s[t]]) * rnorm(1)
  }
  y
}

spec <- CreateSpec(variance.spec = list(model = "sGARCH"),
                   distribution.spec = list(distribution = "norm"),
                   switch.spec = list(K = 2))

set.seed(7)
n_rep <- 40
est <- matrix(NA, n_rep, 8)
se  <- matrix(NA, n_rep, 8)
for (r in 1:n_rep) {
  fit <- FitML(spec, simulate_msgarch(10000))
  invisible(capture.output(s <- summary(fit)))
  e <- s$estimate[, "Estimate"]
  v <- s$estimate[, "Std. Error"]
  # sort regimes by unconditional variance (calm first), as in sim_msgarch11_fit.R
  if (e[1] / (1 - e[2] - e[3]) > e[4] / (1 - e[5] - e[6])) {
    e <- c(e[4:6], e[1:3], 1 - e[8], 1 - e[7])
    v <- c(v[4:6], v[1:3], v[8], v[7])
  }
  est[r, ] <- e
  se[r, ]  <- v
}

out <- data.frame(parameter         = truth$name,
                  true              = truth$value,
                  mean_estimate     = colMeans(est),
                  mc_sd             = apply(est, 2, sd),
                  median_msgarch_se = apply(se, 2, median))
cat(sprintf("Monte Carlo: %d samples of T = 10000\n\n", n_rep))
print(out, digits = 3, row.names = FALSE)
