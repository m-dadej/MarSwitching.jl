using MarSwitching
using DataFrames
using Random
using BenchmarkTools
using StatsBase
using LinearAlgebra
using CSV

# generate artificial data
Random.seed!(1234)

k = 3
μ = [1.0, -0.5, 0.12] 
β = [-1.5, 0.9, 0.0]
β_ns = [0.3333]
σ = [0.3,  0.6, 0.2] 
P = [0.8 0.05 0.2; 0.1 0.85 0.05; 0.1 0.1 0.75]
T = 400

y, _, X = generate_msm(μ, σ, P, T, β = β, β_ns = β_ns)

# using Tables
# CSV.write("artificial.csv", Tables.table([y X]))
df = Matrix(CSV.read("benchmark/artificial.csv", DataFrame))

Random.seed!(1234)
model = MSModel(y, k,
        exog_switching_vars = X[:,2],
        exog_vars = X[:,3])

summary_msm(model)          

mean(abs.(sort([model.β[i][1] for i in 1:model.k]) .- sort(μ)))
mean(abs.(sort([model.β[i][2] for i in 1:model.k]) .- sort(β)))
abs.(model.β[1][end] .- β_ns)
mean(abs.(sort(model.σ) .- sort(σ)))
mean(abs.(sort(diag(model.P)) .- sort(diag(P))))


Random.seed!(123)
@benchmark begin
model = MSModel(y, k, 
        exog_switching_vars = X[:,2],
        exog_vars = X[:,3])
end    

