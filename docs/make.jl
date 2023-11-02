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
    ),
    doctest = false,
    clean = false,
    modules = [MarSwitching],
    pages = ["Home" => "index.md",
             "man/get_started.md",
             "Examples" => "man/example.md",
             "API" => "man/docstrings.md"]
)
deploydocs(
    repo = "github.com/m-dadej/MarSwitching.jl.git",
    devbranch = "main"
)

