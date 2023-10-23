using Documenter
using MarSwitching

makedocs(;
    sitename = "MarSwitching.jl",
    format = Documenter.HTML(),
    doctest = false,
    clean = false,
    modules = [MarSwitching],
    pages = ["Home" => "index.md",
             "man/get_started.md",
             "API" => "man/docstrings.md"]
)

deploydocs(
    repo = "github.com/m-dadej/MarSwitching.jl.git",
    target = "build",
)


