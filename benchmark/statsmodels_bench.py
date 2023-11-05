import timeit
import numpy as np
import pandas as pd
import random
import statsmodels as sm

df = pd.read_csv("artificial.csv")

random.seed(1234)
mod_kns = sm.tsa.MarkovRegression(
            df.iloc[:,0], k_regimes=3, 
            exog = df.iloc[:,2:4],
            switching_exog = [1,0],
            switching_variance=True)
res_kns = mod_kns.fit()
res_kns.summary()
# the parameters are the same as in Julia, so the errors are as well


if __name__ == "__main__":
    df = pd.read_csv("artificial.csv")
    N = 30
    mod_kns = sm.tsa.MarkovRegression(
            df.iloc[:,0], k_regimes=3, 
            exog = df.iloc[:,2:4],
            switching_exog = [1,0],
            switching_variance=True)
    print("Benchmarking...")
    t = timeit.Timer(mod_kns.fit)
    r = t.repeat(N, 1)
    print("Timing: {} s.".format(np.mean(r)))
