# Fits a 2-regime GARCH(1,1) model with R's MSGARCH package (model = "sGARCH") to
# simulated data with known parameters (sim_msgarch11_data.csv, from
# sim_msgarch11_generate.jl) and compares the estimates with the truth.
# sim_msgarch11_fit.jl does the same with MarSwitching.jl, printing an identically
# laid-out table. See README.md for both sets of results.
#
# Requires the MSGARCH package: install.packages("MSGARCH")
#
# Run from anywhere, e.g.:  Rscript benchmark/msarch_vs_msgarch/sim_msgarch11_fit.R

if (!requireNamespace("MSGARCH", quietly = TRUE)) {
  stop("MSGARCH is not installed. Install it first with: install.packages(\"MSGARCH\")")
}
library(MSGARCH)

script_dir <- dirname(sub("--file=", "", commandArgs()[grep("--file=", commandArgs())]))
df     <- read.csv(file.path(script_dir, "sim_msgarch11_data.csv"))
truth  <- read.csv(file.path(script_dir, "sim_msgarch11_params.csv"))
y      <- df$y
n_obs  <- length(y)

spec <- CreateSpec(variance.spec = list(model = "sGARCH"),
                   distribution.spec = list(distribution = "norm"),
                   switch.spec = list(K = 2))

set.seed(42)
t_fit <- system.time(fit <- FitML(spec = spec, data = y))[["elapsed"]]

invisible(capture.output(s <- summary(fit)))   # summary() also prints the full report
tab <- s$estimate
est <- tab[, "Estimate"]
se  <- tab[, "Std. Error"]

# sort regimes by unconditional variance alpha0 / (1 - alpha1 - beta) so that regime 1 is
# the calm one, as in the data-generating process
uncond  <- c(est[["alpha0_1"]] / (1 - est[["alpha1_1"]] - est[["beta_1"]]),
             est[["alpha0_2"]] / (1 - est[["alpha1_2"]] - est[["beta_2"]]))
swapped <- uncond[1] > uncond[2]
if (swapped) {
  # new P_1_1 = old P_2_2 = 1 - old P_2_1; new P_2_1 = old P_1_2 = 1 - old P_1_1
  # (each SE carries over unchanged, since 1 - x has the same SE as x)
  est <- c(est[4:6], est[1:3], 1 - est[["P_2_1"]], 1 - est[["P_1_1"]])
  se  <- c(se[4:6],  se[1:3],  se[["P_2_1"]],      se[["P_1_1"]])
}
names(est) <- names(se) <- truth$name

# share of observations whose most likely smoothed regime is the true one
# (SmoothProb has T + 1 rows; rows 1..T are t = 1..T, the last one is a forecast)
sp     <- State(fit)$SmoothProb[1:n_obs, 1, ]
p_reg2 <- if (swapped) sp[, 1] else sp[, 2]
acc    <- mean((p_reg2 > 0.5) == (df$regime == 2))

cat(sprintf("=== MSGARCH (R, sGARCH, K = 2) on simulated MS-GARCH(1,1), T = %d ===\n", n_obs))
cat(sprintf("fit time: %.1f s\n\n", t_fit))
cat(sprintf("%-10s %10s %10s %10s %12s\n", "parameter", "true", "estimate", "std.err", "(est-true)/se"))
for (i in seq_len(nrow(truth))) {
  cat(sprintf("%-10s %10.4f %10.4f %10.4f %12.2f\n", truth$name[i], truth$value[i],
              est[[i]], se[[i]], (est[[i]] - truth$value[i]) / se[[i]]))
}
cat(sprintf("\nlog-likelihood, MSGARCH convention:       estimate %.4f\n", fit$loglik))
cat(sprintf("regimes correctly classified (smoothed probabilities): %.1f%%\n", 100 * acc))
