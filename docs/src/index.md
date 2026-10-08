# MargLikeGibbsOutput.jl

Estimation of the marginal likelihood from the output of a Gibbs sampler, by the method of

> Chib, S. (1995). Marginal Likelihood from the Gibbs Output.
> *Journal of the American Statistical Association*, 90(432), 1313-1321.

The package works for any number of Gibbs blocks ``\theta_1, \dots, \theta_B``, with or
without latent data, as long as the full conditional posterior of every parameter block
has a known normalizing constant. You supply the full conditionals, the log-likelihood and
the log prior density, and get the estimated log marginal likelihood with its numerical
standard error. The [Method](@ref) page describes the estimator.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/mattiasvillani/MargLikeGibbsOutput.jl")
```

## Quick start

Consider the model ``y_i \sim N(\mu, \sigma^2)`` with independent priors
``\mu \sim N(\mu_0, \tau_0^2)`` and ``\sigma^2 \sim IG(a_0, b_0)``. The Gibbs sampler has
two blocks.

Each full conditional is a function `(state, data, prior) -> distribution`. Here `state`
is a `NamedTuple` with the current value of every block, while `data` and `prior` are the
data and the prior hyperparameters in whatever form you like. The returned distribution
is used both to draw the block and to evaluate its density, so each full conditional is
written only once.

```@example quickstart
using MargLikeGibbsOutput, Distributions, Random

function μ_conditional(state, y, prior)
    precision = 1 / prior.τ₀^2 + length(y) / state.σ²
    return Normal((prior.μ₀ / prior.τ₀^2 + sum(y) / state.σ²) / precision, 1 / sqrt(precision))
end

σ²_conditional(state, y, prior) =
    InverseGamma(prior.a₀ + length(y) / 2, prior.b₀ + sum(abs2, y .- state.μ) / 2)
nothing # hide
```

A [`GibbsModel`](@ref) collects the full conditionals, in the order of the blocks, with
the log-likelihood and the log prior density. Both must include all normalizing constants.

```@example quickstart
model = GibbsModel(
    conditionals = (μ = μ_conditional, σ² = σ²_conditional),
    loglik = (θ, y) -> sum(logpdf.(Normal(θ.μ, sqrt(θ.σ²)), y)),
    logprior = (θ, prior) -> logpdf(Normal(prior.μ₀, prior.τ₀), θ.μ) +
                             logpdf(InverseGamma(prior.a₀, prior.b₀), θ.σ²),
)
nothing # hide
```

[`chib`](@ref) then estimates the log marginal likelihood from the data, the prior
hyperparameters and starting values for the blocks.

```@example quickstart
rng = Xoshiro(1)
y = randn(rng, 50) .+ 1
prior = (μ₀ = 0.0, τ₀ = 10.0, a₀ = 3.0, b₀ = 2.0)
init = (μ = 0.0, σ² = 1.0)

estimate = chib(rng, model, y, prior, init; ndraws = 10_000, burnin = 1_000)
```

The result is a [`ChibEstimate`](@ref) with the estimate, its numerical standard error
and the terms it is built from.

```@example quickstart
estimate.logmarglik, estimate.nse
```

The Gibbs sampler is also available by itself through [`gibbs`](@ref), which returns the
draws of the parameter blocks.

```@example quickstart
draws = gibbs(rng, model, y, prior, init; ndraws = 10_000, burnin = 1_000)
draws[1:3]
```

## Latent data

Models that are sampled with data augmentation give the full conditionals of the latent
variables in `latents`:

```julia
model = GibbsModel(
    conditionals = (β = β_conditional,),
    latents = (z = z_conditional,),
    loglik = ...,
    logprior = ...,
)
```

Three things differ from the parameter blocks:

- The full conditional of a latent variable only needs to support `rand`, since latent
  variables are integrated out rather than evaluated.
- Each sweep updates the latent variables first, so the starting values only need to
  include the parameter blocks.
- `loglik` is the likelihood with the latent variables integrated out.

[Ridge regression](@ref) is an example without latent data. The other examples all have
latent data: [Probit regression](@ref) has one parameter block,
[Finite mixture models](@ref) three, and the [Markov switching model](@ref) draws the
latent states with a custom sampler.

## Choosing the point

The keyword `θstar` of [`chib`](@ref) sets the point ``\theta^*`` at which the marginal
likelihood identity is evaluated. The identity holds at any point, but the estimate is
more precise at a point of high posterior density.

| `θstar` | Point |
|:--|:--|
| [`posteriormean`](@ref) (default) | The posterior mean of each block. |
| [`posteriormode`](@ref) | The draw with the highest posterior density. Use this when the posterior mean may be a low density point or is not a valid parameter value. |
| a function `(draws, logposterior) -> θ*` | Any function of the draws from the full Gibbs run. |
| a `NamedTuple` | A given value for each block. The initial Gibbs run is then skipped. |
