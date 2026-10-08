# MargLikeGibbsOutput.jl

[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://mattiasvillani.github.io/MargLikeGibbsOutput.jl/dev)
[![CI](https://github.com/mattiasvillani/MargLikeGibbsOutput.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/mattiasvillani/MargLikeGibbsOutput.jl/actions/workflows/CI.yml)

Estimation of the marginal likelihood from the output of a Gibbs sampler, by the method of

> Chib, S. (1995). Marginal Likelihood from the Gibbs Output.
> *Journal of the American Statistical Association*, 90(432), 1313-1321.

The package works for any number of Gibbs blocks, with or without latent data, as long as
the full conditional posterior of every parameter block has a known normalizing constant.
You supply the full conditionals, the log-likelihood and the log prior density, and get
the estimated log marginal likelihood with its numerical standard error.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/mattiasvillani/MargLikeGibbsOutput.jl")
```

## Example

Each full conditional is a function `(state, data, prior) -> distribution`. The returned
distribution is used both to draw the block and to evaluate its density, so each full
conditional is written only once.

```julia
using MargLikeGibbsOutput, Distributions

# yᵢ ~ N(μ, σ²) with μ ~ N(μ₀, τ₀²) and σ² ~ IG(a₀, b₀)
function μ_conditional(state, y, prior)
    precision = 1 / prior.τ₀^2 + length(y) / state.σ²
    return Normal((prior.μ₀ / prior.τ₀^2 + sum(y) / state.σ²) / precision, 1 / sqrt(precision))
end

σ²_conditional(state, y, prior) =
    InverseGamma(prior.a₀ + length(y) / 2, prior.b₀ + sum(abs2, y .- state.μ) / 2)

model = GibbsModel(
    conditionals = (μ = μ_conditional, σ² = σ²_conditional),   # blocks θ₁, …, θ_B in order
    loglik = (θ, y) -> sum(logpdf.(Normal(θ.μ, sqrt(θ.σ²)), y)),
    logprior = (θ, prior) -> logpdf(Normal(prior.μ₀, prior.τ₀), θ.μ) +
                             logpdf(InverseGamma(prior.a₀, prior.b₀), θ.σ²),
)

y = randn(50) .+ 1
prior = (μ₀ = 0.0, τ₀ = 10.0, a₀ = 3.0, b₀ = 2.0)
init = (μ = 0.0, σ² = 1.0)

estimate = chib(model, y, prior, init; ndraws = 10_000, burnin = 1_000)
estimate.logmarglik   # estimated log marginal likelihood
estimate.nse          # its numerical standard error
```

## Documentation

The [documentation](https://mattiasvillani.github.io/MargLikeGibbsOutput.jl/dev) describes
the method, latent data and the choice of evaluation point, and has one page for each
example. The pages are generated from the scripts in `examples`:

- `ridge.jl`: ridge regression with a learned regularization parameter, checked against
  numerical integration.
- `nodal.jl`: probit regression with data augmentation (Table 2 of the paper).
- `galaxy.jl`: Gaussian finite mixture models (Table 4).
- `gnp.jl`: Markov switching model for U.S. GNP growth (Table 5).

The scripts for the last two make plots, so run them in the documentation environment:

```
julia --project=docs examples/galaxy.jl
```

To build the documentation locally:

```
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

## A caveat

The estimator assumes that the Gibbs sampler explores the whole posterior. If the sampler
stays in one of several modes, as with label switching in mixture models, the log marginal
likelihood is underestimated and the numerical standard error gives no warning. See

> Neal, R. M. (1999). Erroneous Results in "Marginal Likelihood from the Gibbs Output".

and the mixture and Markov switching examples.
