# Binary probit regression for the nodal involvement data, Section 4.1 of Chib (1995).
# Gibbs sampling with the data augmentation of Albert and Chib (1993): one parameter
# block β and the latent utilities z. Replicates Table 2.

using MargLikeGibbsOutput, Distributions, LinearAlgebra, Random

# Table 1: y, x1 (age), x2 (acid), x3 (xray), x4 (size), x5 (grade)
nodal = [
    0 66 0.48 0 0 0;  0 68 0.56 0 0 0;  0 66 0.50 0 0 0;  0 56 0.52 0 0 0
    0 58 0.50 0 0 0;  0 60 0.49 0 0 0;  0 65 0.46 1 0 0;  0 60 0.62 1 0 0
    1 50 0.56 0 0 1;  0 49 0.55 1 0 0;  0 61 0.62 0 0 0;  0 58 0.71 0 0 0
    0 51 0.65 0 0 0;  1 67 0.67 1 0 1;  0 67 0.47 0 0 1;  0 51 0.49 0 0 0
    0 56 0.50 0 0 1;  0 60 0.78 0 0 0;  0 52 0.83 0 0 0;  0 56 0.98 0 0 0
    0 67 0.52 0 0 0;  0 63 0.75 0 0 0;  1 59 0.99 0 0 1;  0 64 1.87 0 0 0
    1 61 1.36 1 0 0;  1 56 0.82 0 0 0;  0 64 0.40 0 1 1;  0 61 0.50 0 1 0
    0 64 0.50 0 1 1;  0 63 0.40 0 1 0;  0 52 0.55 0 1 1;  0 66 0.59 0 1 1
    1 58 0.48 1 1 0;  1 57 0.51 1 1 1;  1 65 0.49 0 1 0;  0 65 0.48 0 1 1
    0 59 0.63 1 1 1;  0 61 1.02 0 1 0;  0 53 0.76 0 1 0;  0 67 0.95 0 1 0
    0 53 0.66 0 1 1;  1 65 0.84 1 1 1;  1 50 0.81 1 1 1;  1 60 0.76 1 1 1
    1 45 0.70 0 1 1;  1 56 0.78 1 1 1;  1 46 0.70 0 1 0;  1 67 0.67 0 1 0
    1 63 0.82 0 1 0;  1 57 0.67 0 1 1;  1 51 0.72 1 1 0;  1 64 0.89 1 1 0
    1 68 1.26 1 1 1
]
y = nodal[:, 1] .== 1
covariates = (C = ones(length(y)), x1 = nodal[:, 2], logx2 = log.(nodal[:, 3]),
    x3 = nodal[:, 4], x4 = nodal[:, 5], x5 = nodal[:, 6])

# Full conditionals. The prior is β ~ N(a, A⁻¹).
function β_conditional(state, data, prior)
    B = inv(Symmetric(prior.A + data.X'data.X))
    return MvNormal(B * (prior.A * prior.a + data.X'state.z), B)
end

function z_conditional(state, data, prior)
    μ = data.X * state.β
    return product_distribution([y ? truncated(Normal(m), 0, Inf) : truncated(Normal(m), -Inf, 0)
                                 for (y, m) in zip(data.y, μ)])
end

probit = GibbsModel(
    conditionals = (β = β_conditional,),
    latents = (z = z_conditional,),
    loglik = (θ, data) -> sum(logcdf(Normal(), (2y - 1) * η) for (y, η) in zip(data.y, data.X * θ.β)),
    logprior = (θ, prior) -> logpdf(MvNormalCanon(prior.A * prior.a, prior.A), θ.β),
)

# Models and log marginal likelihoods of Table 2
models = [(:C,) => -38.503, (:C, :x1) => -43.175, (:C, :logx2) => -37.916,
    (:C, :x3) => -35.323, (:C, :x4) => -37.234, (:C, :x5) => -39.075,
    (:C, :logx2, :x4) => -36.140, (:C, :logx2, :x3, :x4) => -34.553,
    (:C, :logx2, :x3, :x4, :x5) => -36.233]

rng = Xoshiro(1995)
results = map(models) do (terms, chib1995)
    X = hcat(covariates[terms]...)
    k = size(X, 2)
    prior = (a = fill(0.75, k), A = Matrix(I(k) / 5^2))
    estimate = chib(rng, probit, (; y, X), prior, (β = zeros(k),); ndraws = 5_000, burnin = 500)
    println(rpad(join(terms, " + "), 28), estimate, "   Chib (1995): ", chib1995)
    estimate => chib1995
end
