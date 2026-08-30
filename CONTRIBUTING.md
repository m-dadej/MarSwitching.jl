# Contributing to MarSwitching.jl

Contributions are welcome and appreciated — bug reports, documentation fixes and new
methods alike.

## Reporting a bug

Open an [issue](https://github.com/m-dadej/MarSwitching.jl/issues) that includes:

- the version of MarSwitching.jl and of Julia (`versioninfo()`),
- a minimal example that reproduces the problem — `generate_msm()` is useful for
  producing self-contained data,
- what you expected to happen, and what happened instead.

## Pull requests

- PRs that fix bugs or add new methods are highly appreciated, especially the ones listed
  as planned in the [Functionality](https://github.com/m-dadej/MarSwitching.jl#functionality)
  section of the README.
- Open an issue first if the PR changes the current code substantially, so that the
  approach can be discussed before you invest the work.
- Add tests for new functionality under `test/`, and make sure the existing suite passes:

  ```julia
  julia --project=. -e 'using Pkg; Pkg.test()'
  ```

- Every exported function needs a docstring — the documentation build runs with
  `checkdocs = :exports` and will fail otherwise. To build the docs locally:

  ```julia
  julia --project=docs docs/make.jl
  ```

- If unsure about anything procedural, check the [ColPrac](https://github.com/SciML/ColPrac)
  guide on collaborative practices for Julia packages.

## Questions

For usage questions rather than bugs, the
[Julia Discourse](https://discourse.julialang.org/) is often the better venue, and the
answer there helps the next person with the same question.
