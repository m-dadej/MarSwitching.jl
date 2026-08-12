# Fits a 2-regime ARCH(1) model with R's MSGARCH package (model = "sARCH") on
# weekly S&P 500 returns, for comparison against MSARCHModel() (fit_msarch.jl,
# same folder). See README.md for the reference numbers and how to read the result.
#
# Requires the MSGARCH package: install.packages("MSGARCH")
#
# Run from anywhere, e.g.:  Rscript benchmark/msarch_vs_msgarch/fit_msgarch.R

if (!requireNamespace("MSGARCH", quietly = TRUE)) {
  stop("MSGARCH is not installed. Install it first with: install.packages(\"MSGARCH\")")
}
library(MSGARCH)

script_dir <- dirname(sub("--file=", "", commandArgs()[grep("--file=", commandArgs())]))
df <- read.csv(file.path(script_dir, "spx_weekly_returns.csv"))
y <- df$spx

cat("n obs:", length(y), "\n")
cat("mean(y) =", mean(y), "   sd(y) =", sd(y), "\n\n")

# k = 2 regimes, ARCH(1), zero mean (MSGARCH's variance-only models have no
# mean/location submodel) to match MSARCHModel(y, 2, 1, intercept = "no").
spec <- CreateSpec(variance.spec = list(model = "sARCH"),
                    distribution.spec = list(distribution = "norm"),
                    switch.spec = list(K = 2))

set.seed(42)
fit <- FitML(spec = spec, data = y)

p <- fit$par
P_1_1 <- p[["P_1_1"]]
P_2_2 <- 1 - p[["P_2_1"]]

cat("=== MSGARCH fit (sARCH, K = 2) ===\n")
cat("omega       = (", p[["alpha0_1"]], ",", p[["alpha0_2"]], ")\n")
cat("alpha       = (", p[["alpha1_1"]], ",", p[["alpha1_2"]], ")\n")
cat("P (diag)    = (", P_1_1, ",", P_2_2, ")\n")
cat("expected duration (weeks) = (", 1 / (1 - P_1_1), ",", 1 / (1 - P_2_2), ")\n")
cat("Likelihood  =", fit$loglik, "\n")

print(fit)
