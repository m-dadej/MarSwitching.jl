library(MSwM)
library(microbenchmark)


df <- read.csv("C:/Users/HP/Desktop/jstatsoft/benchmark/artificial.csv")

mod <- lm(`Column1` ~ `Column3` + `Column4`, data = df)

mod.mswm <- msmFit(mod,k=3,p=0,sw=c(TRUE,TRUE, FALSE, TRUE),control=list(parallel=FALSE))

summary(mod.mswm)

mean(abs((sort(mod.mswm@Coef$`(Intercept)`) - sort(c(1.0, -0.5, 0.12)))))
mean(abs((sort(mod.mswm@Coef$`Column3`) - sort(c(-1.5, 0.9, 0.0)))))
abs(mod.mswm@Coef$`Column4`[1] - 0.333)
mean(abs(sort(mod.mswm@std) - sort(c(0.3,  0.6, 0.2))))
mean(abs(sort(diag(mod.mswm@transMat)) - sort(c(0.8, 0.85, 0.75))))


mswm_fit <- function(df){
  mod <- lm(`Column1` ~ `Column3` + `Column4`, data = df)
  
  mod.mswm <- msmFit(mod,k=3,p=0,sw=c(TRUE,TRUE, FALSE, TRUE),control=list(parallel=FALSE))
  return(mod)  
}

mod <- mswm_fit(df)

mb <- microbenchmark("mswm" = {mod <- mswm_fit(df)})
