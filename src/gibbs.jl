"""
    GibbsModel(; conditionals, loglik, logprior, latents = (;))

A model whose posterior is simulated by Gibbs sampling over the parameter blocks
`θ₁, …, θ_B` and, with data augmentation, the latent variables `z`.

- `conditionals`: `NamedTuple` with one full conditional posterior per parameter block,
  in the order `θ₁, …, θ_B`. Each is a function `(state, data, prior) -> distribution`
  where the returned object supports `rand` and a properly normalized `logpdf`,
  for example any distribution from Distributions.jl.
- `latents`: `NamedTuple` with the full conditionals of the latent variables. Same
  signature, but the returned object only needs to support `rand`.
- `loglik`: the log-likelihood `(θ, data) -> Real`, with the latent variables integrated out.
- `logprior`: the log prior density `(θ, prior) -> Real`.

`state` is a `NamedTuple` with the current value of every block and latent variable, and
`θ` one with the parameter blocks only. Both log densities must include all normalizing
constants.
"""

"""Container for Gibbs-sampling conditionals and model density functions.

`@kwdef` generates a keyword-based constructor. The type parameters capture the
concrete types of the conditionals, latent conditionals, likelihood, and prior.
"""
Base.@kwdef struct GibbsModel{C<:NamedTuple,L<:NamedTuple,F,P}
    conditionals::C
    latents::L = (;)
    loglik::F
    logprior::P
end

# One Gibbs sweep: update each block in turn from its full conditional.
function sweep(rng, conditionals, state, data, prior)
    for (name, conditional) in pairs(conditionals)
        state = merge(state, (; name => rand(rng, conditional(state, data, prior))))
    end
    return state
end

"""
    gibbs(f, rng, conditionals, data, prior, init; ndraws, burnin = 0)

Gibbs sampling from the full `conditionals`, starting at the state `init`. Entries of
`init` without a conditional are held fixed, which is what makes the same sampler
usable for Chib's reduced runs. Returns `f(state)` for each of the `ndraws` states
after `burnin`.
"""
function gibbs(f, rng::AbstractRNG, conditionals::NamedTuple, data, prior, init::NamedTuple;
        ndraws, burnin = 0)
    state = init
    for _ in 1:burnin
        state = sweep(rng, conditionals, state, data, prior)
    end
    return map(1:ndraws) do _
        state = sweep(rng, conditionals, state, data, prior)
        f(state)
    end
end

"""
    gibbs([rng], model::GibbsModel, data, prior, init; ndraws, burnin = 0)

Posterior draws of the parameter blocks of `model`. Each sweep updates the latent
variables first, so `init` only needs to hold starting values for the parameter blocks.
"""
function gibbs(rng::AbstractRNG, model::GibbsModel, data, prior, init::NamedTuple; kwargs...)
    blocks = keys(model.conditionals)
    return gibbs(state -> state[blocks], rng, merge(model.latents, model.conditionals),
        data, prior, init; kwargs...)
end

gibbs(model::GibbsModel, args...; kwargs...) = gibbs(default_rng(), model, args...; kwargs...)
