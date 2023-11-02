
## Regime switching Phillips curve

One of the most popular macroeconomic relationships is the trade-off between inflation and unemployment. The so-called Phillips curve is discussed in both introductory macroeconomics courses and at meetings of central banks.

However, many policymakers and academic economists have argued that the historical relationship has changed over time. The 'flattening' of the Phillips curve poses a challenge for policymakers, as it can imply that countercyclical policy may not be effective in steering inflation toward the established central bank's target.

To investigate the time-varying nature of the Phillips curve, we estimate a Markov switching model.

First we would need a dataset with inflation and unemployment. We will use the data from the Federal Reserve Bank of St. Louis (FRED) database. The data are available in the repo of the package.

```@example
using MarSwitching
using DataFrames
using CSV

df = CSV.read("my_assets/philips.csv", DataFrame)
```

Let's see how the relationship looks like in the data:

```@example
using Plots
plot(df.cpi_ch, df.unemp, zcolor = (df.date .> DateTime("2000-01-01")),
     c = palette([:coral, :steelblue], 7), seriestype=:scatter, legend = :none,
     xlabel = "CPI change", ylabel = "Unemployment rate", title = "Phillips curve for XX and XXI (red) century ")
```
![Plot](my_assets/philips.png)

Overall, the relationship is far from being clear. However, we may see that indeed the relationship in the XX century seems to be negative, at least in a non-linear way. One cannot say it about the current century, when the data is very much random (not to mention the evident autocorrelation). 


Now, how can theory guide our model specification? The developments in New Keynesian economic theory, provides a model where the

