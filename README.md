# MargLikeGibbsOutput.jl

Marginal likelihood from the Gibbs output, following

> Chib, S. (1995). Marginal Likelihood from the Gibbs Output.
> *Journal of the American Statistical Association*, 90(432), 1313-1321.

The estimator works for any number of Gibbs blocks `θ₁, …, θ_B`, with or without latent
data `z`, as long as every full conditional posterior of the parameter blocks has a
known normalizing constant.

## The method

The basic marginal likelihood identity holds at any point `θ*`:

    ln m(y) = ln f(y|θ*) + ln π(θ*) - ln π(θ*|y)

The posterior ordinate is decomposed as

    π(θ*|y) = π(θ₁*|y) π(θ₂*|y, θ₁*) ⋯ π(θ_B*|y, θ₁*, …, θ_{B-1}*)

and factor `r` is estimated by averaging the full conditional density
`π(θᵣ*|y, θ₁*, …, θᵣ₋₁*, θᵣ₊₁, …, θ_B, z)` over the draws from a *reduced* Gibbs run in
which the blocks `θ₁, …, θᵣ₋₁` are held fixed at `θ*`. A reduced run is the same Gibbs
sampler with the conditionals of the fixed blocks left out, so no code beyond the full
conditionals is needed.

## Usage

A model is specified by its full conditional posteriors, the log-likelihood and the log
prior density. Each full conditional is a function `(state, data, prior) -> distribution`
where `state` is a `NamedTuple` with the current value of all blocks and latent
variables, `data` is the data and `prior` the prior hyperparameters, both in whatever
form you like. The returned distribution is used both to draw the block (`rand`) and to
evaluate its ordinate (`logpdf`), so each conditional is written only once.

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

`chib` makes one run of the full Gibbs sampler to choose `θ*` and then one run per block
to estimate the ordinates, each with `ndraws` draws after `burnin`. A run with nothing
left to integrate out (the last block in a model without latent data) is skipped, since
the full conditional is then the exact ordinate.

### Latent data

Latent variables are given by their full conditionals in `latents`. These only need to
support `rand`, since latent variables are integrated out rather than evaluated. Each
sweep updates the latent variables first, so `init` only needs starting values for the
parameter blocks. `loglik` is the likelihood with the latent variables integrated out.

```julia
probit = GibbsModel(
    conditionals = (β = β_conditional,),
    latents = (z = z_conditional,),
    loglik = ...,
    logprior = ...,
)
```

### The point θ*

The keyword `θstar` sets the point at which the identity is evaluated:

- `θstar = posteriormean` (default): the posterior mean of each block.
- `θstar = posteriormode`: the draw with the highest posterior density. Use this when the
  posterior mean may be a low density point or is not a valid parameter value.
- any function `(draws, logposterior) -> θ*` of the draws from the full Gibbs run.
- a `NamedTuple` with a value for each block, in which case the initial run is skipped.

### Numerical standard error

The standard error follows Section 3 of the paper. With `h⁽ᵍ⁾` the vector of the `B`
full conditional ordinates at draw `g`, the variance of their average `ĥ` is estimated
with the Newey-West estimator using `lags` lags (10 by default, as in the paper), and
the delta method gives the variance of `∑ᵣ ln ĥᵣ`.

### Gibbs sampling

The sampler is also available by itself, returning a vector with the draws of the
parameter blocks:

```julia
draws = gibbs(model, y, prior, init; ndraws = 10_000, burnin = 1_000)
```

## Examples

The `examples` folder replicates the applications in the paper:

- `nodal.jl`: binary probit regression with data augmentation (Table 2).
- `galaxy.jl`: Gaussian finite mixture models with three blocks and latent component
  indicators (Table 4).
- `gnp.jl`: Markov switching model for U.S. GNP growth, with the latent states drawn by
  forward filtering and backward sampling (Table 5).

## A caveat

The estimator assumes that the Gibbs sampler explores the whole posterior. A mixture
model with `d` components has `d!` symmetric modes that differ only in the labelling of
the components. If the sampler stays in one of them, the posterior ordinate is
overestimated by a factor `d!` and the log marginal likelihood is too low by `ln d!`; see

> Neal, R. M. (1999). Erroneous Results in "Marginal Likelihood from the Gibbs Output".

`examples/galaxy.jl` shows this: the estimates there differ by `ln d!` from brute-force
averages of the likelihood over prior draws.
