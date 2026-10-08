# Gaussian finite mixture models for the galaxy data, Section 4.2.1 of Chib (1995).
# Three parameter blocks (μ, σ², q) and the latent component indicators z, so the
# posterior ordinate needs two reduced Gibbs runs. Replicates Table 4.
#
# The Gibbs sampler does not move between the d! modes that differ only in the labelling
# of the components, so the posterior ordinate is overestimated by a factor d! and the
# log marginal likelihood is too low by ln d!, as pointed out by Neal (1999, Erroneous
# Results in "Marginal Likelihood from the Gibbs Output"). The first two rows of Table 4
# are reproduced; the third is the entry Neal showed to be wrong. Adding ln d! gives
# -239.75, -226.83 and -226.82, in agreement with brute-force averages of the likelihood
# over 10⁸ prior draws: -239.76, -226.84 and -226.80.

using MargLikeGibbsOutput, Distributions, LinearAlgebra, Random
using Distributions: logfactorial

# Table 3: velocities (km/second) of 82 galaxies, in units of 1000 km/second
y = [9172, 9350, 9483, 9558, 9775, 10227, 10406, 16084, 16170, 18419, 18552, 18600,
    18927, 19052, 19070, 19330, 19343, 19349, 19440, 19473, 19529, 19541, 19547, 19663,
    19846, 19856, 19863, 19914, 19918, 19973, 19989, 20166, 20175, 20179, 20196, 20215,
    20221, 20415, 20629, 20795, 20821, 20846, 20875, 20986, 21137, 21492, 21701, 21814,
    21921, 21960, 22185, 22209, 22242, 22249, 22314, 22374, 22495, 22746, 22747, 22888,
    22914, 23206, 23241, 23263, 23484, 23538, 23542, 23666, 23706, 23711, 24129, 24285,
    24289, 24366, 24717, 24990, 25633, 26960, 26995, 32065, 32789, 34279] / 1000

# Prior: μⱼ ~ N(μ₀, 1/A), σⱼ² ~ IG(ν₀/2, δ₀/2) and q ~ Dirichlet(α).
prior(d) = (μ₀ = 20.0, A = 1 / 100, ν₀ = 6.0, δ₀ = 40.0, α = ones(d))

# σ² is a vector with one variance per component, or a scalar when they are restricted
# to be equal.
variances(σ², d) = σ² .* ones(d)
members(z, j) = findall(==(j), z)

function μ_conditional(state, y, prior)
    d = length(state.q)
    σ² = variances(state.σ², d)
    n = [count(==(j), state.z) for j in 1:d]
    B = @. 1 / (prior.A + n / σ²)
    μ̂ = [B[j] * (prior.A * prior.μ₀ + sum(y[members(state.z, j)]) / σ²[j]) for j in 1:d]
    return MvNormal(μ̂, Diagonal(B))
end

δ(state, y, j) = sum(abs2, y[members(state.z, j)] .- state.μ[j])

σ²_conditional(state, y, prior) = product_distribution(
    [InverseGamma((prior.ν₀ + count(==(j), state.z)) / 2, (prior.δ₀ + δ(state, y, j)) / 2)
     for j in eachindex(state.μ)])

σ²_conditional_equal(state, y, prior) = InverseGamma((prior.ν₀ + length(y)) / 2,
    (prior.δ₀ + sum(j -> δ(state, y, j), eachindex(state.μ))) / 2)

q_conditional(state, y, prior) =
    Dirichlet(prior.α .+ [count(==(j), state.z) for j in eachindex(state.q)])

function z_conditional(state, y, prior)
    components = Normal.(state.μ, sqrt.(variances(state.σ², length(state.q))))
    return product_distribution([Categorical(normalize(state.q .* pdf.(components, yᵢ), 1)) for yᵢ in y])
end

loglik(θ, y) = loglikelihood(
    MixtureModel(Normal.(θ.μ, sqrt.(variances(θ.σ², length(θ.q)))), θ.q), y)

logprior(θ, prior) = sum(logpdf.(Normal(prior.μ₀, 1 / sqrt(prior.A)), θ.μ)) +
    sum(logpdf.(InverseGamma(prior.ν₀ / 2, prior.δ₀ / 2), θ.σ²)) + logpdf(Dirichlet(prior.α), θ.q)

mixture(σ²_conditional) = GibbsModel(
    conditionals = (μ = μ_conditional, σ² = σ²_conditional, q = q_conditional),
    latents = (z = z_conditional,); loglik, logprior)

init(d, σ²) = (μ = collect(range(10, 30, d)), σ² = σ², q = fill(1 / d, d))

# Models and log marginal likelihoods of Table 4
models = [
    ("Two components, equal variances", mixture(σ²_conditional_equal), init(2, 4.0), -240.464),
    ("Three components, equal variances", mixture(σ²_conditional_equal), init(3, 4.0), -228.620),
    ("Three components, unrestricted", mixture(σ²_conditional), init(3, fill(4.0, 3)), -224.138),
]

rng = Xoshiro(1995)
for (name, model, start, chib1995) in models
    d = length(start.q)
    estimate = chib(rng, model, y, prior(d), start; ndraws = 5_000, burnin = 500, θstar = posteriormode)
    println(rpad(name, 36), estimate, "   Chib (1995): ", chib1995,
        "   corrected for label switching: ", round(estimate.logmarglik + logfactorial(d); digits = 3))
end
