# Speed benchmark for MSGARCH's FitML() (model = "sARCH", K = 2) on the same
# fit used in fit_msgarch.jl / fit_msarch.jl (see README.md, "Speed comparison"
# section). R has no JIT compilation step, so there is no cold/warm distinction
# to make here -- these are just repeated single optimization runs, matching
# FitML()'s default of one run from one starting point.
#
# Run from anywhere, e.g.:  Rscript benchmark/msarch_vs_msgarch/speed_msgarch.R

if (!requireNamespace("MSGARCH", quietly = TRUE)) {
  stop("MSGARCH is not installed. Install it first with: install.packages(\"MSGARCH\")")
}
library(MSGARCH)

script_dir <- dirname(sub("--file=", "", commandArgs()[grep("--file=", commandArgs())]))
df <- read.csv(file.path(script_dir, "spx_weekly_returns.csv"))
y <- df$spx

spec <- CreateSpec(variance.spec = list(model = "sARCH"),
                    distribution.spec = list(distribution = "norm"),
                    switch.spec = list(K = 2))

N <- 20
times <- numeric(N)
set.seed(42)
for (i in 1:N) {
  t0 <- Sys.time()
  fit <- FitML(spec = spec, data = y)
  times[i] <- as.numeric(Sys.time() - t0, units = "secs")
}

cat("calls, N =", N, "(default effort: 1 optimization run)\n")
cat("  mean   =", round(mean(times), 4), "s\n")
cat("  median =", round(median(times), 4), "s\n")
cat("  min    =", round(min(times), 4), "s\n")
cat("  max    =", round(max(times), 4), "s\n")
cat("\nLikelihood =", fit$loglik, "\n")
