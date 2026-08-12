using MarSwitching
using DataFrames
using CSV
using Plots
using Random
using Dates

df = CSV.read("C:/Users/HP/Downloads/philips.csv", DataFrame, missingstring = "NA")

model_df = dropmissing(select(df, [:cpi, :unemp,  :infexp, :s_shock])) 

plot_df = filter(x -> x.cpi .> -2, model_df) # remove outliers

phil_plot = plot(plot_df.unemp, plot_df.cpi,
                seriestype=:scatter, legend = :none,
                title = "Phillips curve",
                xlabel = "Unemployment rate gap", ylabel = "CPI change")

savefig(phil_plot, "docs/src/man/examples/my_assets/philips.svg")

x = [ones(size(model_df)[1]) model_df.unemp]
(x'x)^(-1)*(x'model_df.cpi)

x = [ones(size(model_df)[1]) model_df.unemp model_df.infexp]
(x'x)^(-1)*(x'model_df.cpi)

x = [ones(size(model_df)[1]) model_df.unemp model_df.infexp model_df.s_shock]
(x'x)^(-1)*(x'model_df.cpi)


Random.seed!(0)
model = MSModel(model_df.cpi, 2, 
                exog_switching_vars = [model_df.unemp model_df.infexp],
                exog_vars =  model_df.s_shock)

summary_msm(model)

probs_phil = plot(df.date[9:end], 
                [filtered_probs(model)[:,1] filtered_probs(model)[:,2]],
                xticks = (Date.(minimum(year.(df.date)):4:maximum(year.(df.date))),minimum(year.(df.date)):4:maximum(year.(df.date))),
                xrotation= 45,
                label     = ["Steep Phillips curve" "Flat Phillips curve"],
                linewidth = 2,
                legend = :bottomleft) 

savefig(probs_phil, "docs/src/man/examples/my_assets/probs_phil.svg")

