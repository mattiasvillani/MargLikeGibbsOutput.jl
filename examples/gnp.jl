# Markov mixture (Markov switching) model for U.S. GNP growth, Section 4.2.2 of Chib (1995).
# Three parameter blocks (μ, σ², q) and the latent states z, which are drawn jointly by
# forward filtering and backward sampling. Compare with Table 5.
#
#   yₜ | zₜ = j ~ N(μⱼ, σ²),    Pr(zₜ = j | zₜ₋₁ = i) = pᵢⱼ,    Pr(z₁ = j) = π₁ⱼ
#
# This is not an exact replication. The paper does not give the data or π₁; here the data
# are the March 1994 vintage of real GNP from ALFRED (series GNPC96) and π₁ is uniform.
#
# The posterior has two modes, the one in Table 5 with μ₁ < μ₂ and a smaller one with the
# regimes interchanged, which the prior does not rule out. The Gibbs sampler moves between
# them only rarely, so how much of each it visits varies from run to run, and with it the
# estimate, by more than the numerical standard error suggests. A run that stays in the
# main mode gives about -230.2, while a brute-force average of the likelihood over
# 2·10⁷ prior draws gives -229.76.

using MargLikeGibbsOutput, Distributions, LinearAlgebra, Random, Statistics
using DelimitedFiles: readdlm

# Quarterly growth rates of real GNP in percent, 1951.2 to 1992.4
gnp = vec(readdlm(joinpath(@__DIR__, "gnp.csv"), ',', skipstart = 1)[:, 2])
y = 100 * diff(log.(gnp))

# Prior: μⱼ ~ N(μ₀ⱼ, τ₀²), σ² ~ IG(a₀, b₀) and qᵢ ~ Dirichlet(αᵢ), where qᵢ is row i of the
# transition matrix P. Both the hyperparameters α and the block q hold qᵢ in column i.
prior = (μ₀ = [0.0, 0.75], τ₀² = 2.0, a₀ = 4.0, b₀ = 4.0, α = [4.0 1.0; 1.0 4.0], π₁ = [0.5, 0.5])

# Filtered state probabilities Pr(zₜ = j | y₁, …, yₜ, θ) in row t, and the log-likelihood,
# which is the sum of the log one-step-ahead prediction densities.
function forwardfilter(θ, y, π₁)
    filtered = Matrix{Float64}(undef, length(y), length(π₁))
    predicted, loglik = π₁, 0.0
    for (t, yₜ) in enumerate(y)
        joint = predicted .* pdf.(Normal.(θ.μ, sqrt(θ.σ²)), yₜ)
        loglik += log(sum(joint))
        filtered[t, :] = joint / sum(joint)
        predicted = θ.q * filtered[t, :]
    end
    return filtered, loglik
end

# The joint distribution of the states given the data and parameters. Being latent, it
# only needs `rand`, which samples the states backwards from the filtered probabilities.
struct StatesPosterior
    filtered::Matrix{Float64}
    q::Matrix{Float64}
end

function Base.rand(rng::AbstractRNG, d::StatesPosterior)
    n = size(d.filtered, 1)
    z = Vector{Int}(undef, n)
    z[n] = rand(rng, Categorical(d.filtered[n, :]))
    for t in n-1:-1:1
        z[t] = rand(rng, Categorical(normalize(d.filtered[t, :] .* d.q[z[t+1], :], 1)))
    end
    return z
end

z_conditional(state, y, prior) = StatesPosterior(first(forwardfilter(state, y, prior.π₁)), state.q)

function μ_conditional(state, y, prior)
    regimes = eachindex(prior.μ₀)
    B = [1 / (1 / prior.τ₀² + count(==(j), state.z) / state.σ²) for j in regimes]
    μ̂ = [B[j] * (prior.μ₀[j] / prior.τ₀² + sum(y[state.z .== j]) / state.σ²) for j in regimes]
    return MvNormal(μ̂, Diagonal(B))
end

σ²_conditional(state, y, prior) = InverseGamma(prior.a₀ + length(y) / 2,
    prior.b₀ + sum(abs2, y - state.μ[state.z]) / 2)

function q_conditional(state, y, prior)
    transitions = [count(==((i, j)), zip(state.z, state.z[2:end])) for j in axes(prior.α, 1), i in axes(prior.α, 2)]
    return product_distribution(Dirichlet.(eachcol(prior.α + transitions)))
end

markovmixture = GibbsModel(
    conditionals = (μ = μ_conditional, σ² = σ²_conditional, q = q_conditional),
    latents = (z = z_conditional,),
    loglik = (θ, y) -> last(forwardfilter(θ, y, prior.π₁)),
    logprior = (θ, prior) -> sum(logpdf.(Normal.(prior.μ₀, sqrt(prior.τ₀²)), θ.μ)) +
        logpdf(InverseGamma(prior.a₀, prior.b₀), θ.σ²) +
        logpdf(product_distribution(Dirichlet.(eachcol(prior.α))), θ.q),
)

rng = Xoshiro(1995)
init = (μ = [0.0, 1.0], σ² = 1.0, q = [0.8 0.2; 0.2 0.8])
draws = gibbs(rng, markovmixture, y, prior, init; ndraws = 60_000, burnin = 600)
estimate = chib(rng, markovmixture, y, prior, init; ndraws = 6_000, burnin = 600, θstar = posteriormode)

# Table 5: posterior means and standard deviations in the mode with μ₁ < μ₂, and the log
# marginal likelihood
table5 = ["μ₁" => (d -> d.μ[1], -0.313, 0.314), "μ₂" => (d -> d.μ[2], 1.038, 0.111),
    "σ²" => (d -> d.σ², 0.672, 0.089), "p₁₁" => (d -> d.q[1, 1], 0.743, 0.098),
    "p₂₂" => (d -> d.q[2, 2], 0.911, 0.042)]
for (name, (parameter, chibmean, chibstd)) in table5
    x = parameter.(filter(d -> d.μ[1] < d.μ[2], draws))
    println(rpad(name, 5), "mean = ", round(mean(x); digits = 3), ", std = ", round(std(x); digits = 3),
        "   Chib (1995): ", chibmean, ", ", chibstd)
end
println("Share of draws with μ₁ < μ₂: ", round(mean(d -> d.μ[1] < d.μ[2], draws); digits = 2))
println(estimate, "   Chib (1995): -229.496, NSE = 0.028")
