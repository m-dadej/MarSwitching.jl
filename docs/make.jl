push!(LOAD_PATH,"../src/")

using Documenter
using MarSwitching
using DocThemeIndigo

# 1. generate the indigo theme css
indigo = DocThemeIndigo.install(MarSwitching)

DocThemeIndigo.install
makedocs(;
    sitename = "MarSwitching.jl",
    format=Documenter.HTML(;
        assets=String[indigo],
        canonical="https://m-dadej.github.io/MarSwitching.jl/stable",
        description="Estimate Markov switching (regime switching) dynamic regression models " *
                    "in Julia: k regimes, time-varying transition probabilities, MS-ARCH, " *
                    "filtered and smoothed probabilities.",
    ),
    doctest = false,
    clean = false,
    modules = [MarSwitching],
    checkdocs = :exports,
    pages = ["Markov switching models in Julia" => "index.md",
             "man/get_started.md",
             "Examples" => Any["man/examples/example.md",
                               "man/examples/example_spx.md"],
             "man/comparison.md",
             "API reference" => "man/docstrings.md"]
)
deploydocs(
    repo = "github.com/m-dadej/MarSwitching.jl.git",
    devbranch = "main"
)
