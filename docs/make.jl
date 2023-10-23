push!(LOAD_PATH,"../src/")

using Documenter
using MarSwitching
using DocumenterTools

makedocs(;
    sitename = "MarSwitching.jl",
    doctest = false,
    clean = false,
    modules = [MarSwitching],
    pages = ["Home" => "index.md",
             "man/get_started.md",
             "API" => "man/docstrings.md"]
)
deploydocs(
    repo = "github.com/m-dadej/MarSwitching.jl.git", 
    devbranch = "gh-pages"
)

