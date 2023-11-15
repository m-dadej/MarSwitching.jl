import timeit
import numpy as np
import pandas as pd
import random
import statsmodels.api as sm

df = pd.read_csv("benchmark/artificial.csv")

random.seed(1234)
mod_ms = sm.tsa.MarkovRegression(
            df.iloc[:,0], k_regimes=3, 
            exog = df.iloc[:,2:4],
            switching_exog = [1,0],
            switching_variance=True)
res_ms = mod_ms.fit()

res_ms.summary()
# the parameters are the same as in Julia, so the errors are as well


if __name__ == "__main__":
    df = pd.read_csv("artificial.csv")
    N = 30
    mod_ms = sm.tsa.MarkovRegression(
            df.iloc[:,0], k_regimes=3, 
            exog = df.iloc[:,2:4],
            switching_exog = [1,0],
            switching_variance=True)
    print("Benchmarking...")
    t = timeit.Timer(mod_ms.fit)
    r = t.repeat(N, 1)
    print("Timing: {} s.".format(np.mean(r)))
