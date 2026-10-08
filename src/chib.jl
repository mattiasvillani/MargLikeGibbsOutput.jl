"""
    ChibEstimate

Chib's (1995) estimate of the log marginal likelihood from the basic marginal likelihood
identity `ln m(y) = ln f(y|θ*) + ln π(θ*) - ln π(θ*|y)`.

# Fields
- `logmarglik`: estimated log marginal likelihood.
- `nse`: numerical standard error of `logmarglik`.
- `loglik`, `logprior`: log-likelihood and log prior density at `θstar`.
- `logordinates`: estimated log reduced conditional ordinates `ln π(θᵣ*|y, θ₁*, …, θᵣ₋₁*)`,
  one per block. Their sum is the log posterior ordinate `ln π(θ*|y)`.
- `θstar`: the point `θ*` at which the identity is evaluated.
"""
struct ChibEstimate{T<:NamedTuple,O<:NamedTuple}
    logmarglik::Float64
    nse::Float64
    loglik::Float64
    logprior::Float64
    logordinates::O
    θstar::T
end

Base.show(io::IO, e::ChibEstimate) = print(io, "ChibEstimate(log marginal likelihood = ",
    round(e.logmarglik; digits = 3), ", NSE = ", round(e.nse; sigdigits = 2), ")")

"""
    posteriormean(draws, logposterior)

Posterior mean of each parameter block. A choice for the `θstar` argument of [`chib`](@ref).
"""
function posteriormean(draws, _)
    blocks = keys(first(draws))
    return NamedTuple{blocks}(map(block -> mean(draw[block] for draw in draws), blocks))
end

"""
    posteriormode(draws, logposterior)

The draw with the highest posterior density. A choice for the `θstar` argument of
[`chib`](@ref).
"""
posteriormode(draws, logposterior) = argmax(logposterior, draws)

"""
    chib([rng], model::GibbsModel, data, prior, init;
         ndraws = 10_000, burnin = 1_000, lags = 10, θstar = posteriormean)

Estimate the log marginal likelihood of `model` by the method of Chib (1995).

The posterior ordinate is decomposed as `π(θ*|y) = ∏ᵣ π(θᵣ*|y, θ₁*, …, θᵣ₋₁*)` and each
factor is estimated by averaging the full conditional density of block `r` at `θᵣ*` over
the draws from a reduced Gibbs run in which blocks `1, …, r-1` are held fixed at `θ*`.
Each run uses `ndraws` draws after `burnin`.

`init` is a `NamedTuple` with starting values for the parameter blocks. The point `θstar`
is either a `NamedTuple` with a value for each block or a function
`(draws, logposterior) -> θ*` applied to the draws from an initial run of the full Gibbs
sampler, such as [`posteriormean`](@ref) or [`posteriormode`](@ref).

The numerical standard error follows Section 3 of the paper: the delta method applied to
a Newey-West estimate, with `lags` lags, of the variance of the estimated ordinates.

Returns a [`ChibEstimate`](@ref).
"""
function chib(rng::AbstractRNG, model::GibbsModel, data, prior, init::NamedTuple;
        ndraws = 10_000, burnin = 1_000, lags = 10, θstar = posteriormean)
    θstar = evaluationpoint(θstar, rng, model, data, prior, init; ndraws, burnin)
    logh = logconditionals(rng, model, data, prior, merge(init, θstar); ndraws, burnin)
    logordinates = vec(logsumexp(logh; dims = 1)) .- log(ndraws)

    loglik = model.loglik(θstar, data)
    logprior = model.logprior(θstar, prior)
    # Delta method: ∑ᵣ ln ĥᵣ has gradient 1/ĥᵣ, so its variance is that of the mean of ∑ᵣ hᵣ/ĥᵣ.
    relative = vec(sum(exp.(logh .- logordinates'); dims = 2))
    nse = sqrt(longrunvariance(relative, lags) / ndraws)

    blocks = keys(model.conditionals)
    return ChibEstimate(loglik + logprior - sum(logordinates), nse, loglik, logprior,
        NamedTuple{blocks}(logordinates), θstar[blocks])
end

chib(model::GibbsModel, args...; kwargs...) = chib(default_rng(), model, args...; kwargs...)

evaluationpoint(θstar::NamedTuple, args...; kwargs...) = θstar
function evaluationpoint(point, rng, model, data, prior, init; kwargs...)
    draws = gibbs(rng, model, data, prior, init; kwargs...)
    return point(draws, θ -> model.loglik(θ, data) + model.logprior(θ, prior))
end

# Column r holds ln π(θᵣ*|y, θ₁*, …, θᵣ₋₁*, θᵣ₊₁, …, θ_B, z) over the draws from the
# reduced run that holds blocks 1, …, r-1 fixed at θ*.
function logconditionals(rng, model, data, prior, start; ndraws, burnin)
    blocks = keys(model.conditionals)
    logh = Matrix{Float64}(undef, ndraws, length(blocks))
    for (r, block) in enumerate(blocks)
        free = merge(model.latents, model.conditionals[blocks[r:end]])
        # With only block r left to sample, its conditional is the exact ordinate.
        n, b = length(free) == 1 ? (1, 0) : (ndraws, burnin)
        logh[:, r] .= gibbs(rng, free, data, prior, start; ndraws = n, burnin = b) do state
            logpdf(model.conditionals[block](state, data, prior), start[block])
        end
    end
    return logh
end

# Newey-West estimate of 2πS(0), the spectral density of x at frequency zero.
function longrunvariance(x, lags)
    e = x .- mean(x)
    γ(s) = dot(view(e, 1+s:length(e)), view(e, 1:length(e)-s)) / length(e)
    return γ(0) + 2 * sum(s -> (1 - s / (lags + 1)) * γ(s), 1:lags; init = 0.0)
end
