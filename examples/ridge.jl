# # Ridge regression
#
# This example estimates the marginal likelihood of a linear regression with an
# L2-regularization (ridge) prior where the amount of regularization is learned from the
# data. The Gibbs sampler is the one in Box 16.2 of Villani's *Bayesian Learning* book,
# and the notation follows the book. The model and data are from Computer Lab 4 of the
# course Advanced Bayesian Learning.
#
# In this model the marginal likelihood can also be computed by numerical integration,
# so the example ends with a check of the estimate.

using MargLikeGibbsOutput, Distributions, LinearAlgebra, Random, Statistics
using Distributions: loggamma

# ## Data
#
# The prostate cancer data from *The Elements of Statistical Learning* (Hastie,
# Tibshirani and Friedman, 2009, Section 3.2.1), with all 97 observations. The response
# is `lpsa` in the last column, and the covariates are `lcavol`, `lweight`, `age`,
# `lbph`, `svi`, `lcp`, `gleason` and `pgg45`.

prostate = [
    -0.579818 2.769459 50 -1.386294 0 -1.386294 6 0 -0.430783
    -0.994252 3.319626 58 -1.386294 0 -1.386294 6 0 -0.162519
    -0.510826 2.691243 74 -1.386294 0 -1.386294 7 20 -0.162519
    -1.203973 3.282789 58 -1.386294 0 -1.386294 6 0 -0.162519
    0.751416 3.432373 62 -1.386294 0 -1.386294 6 0 0.371564
    -1.049822 3.228826 50 -1.386294 0 -1.386294 6 0 0.765468
    0.737164 3.473518 64 0.615186 0 -1.386294 6 0 0.765468
    0.693147 3.539509 58 1.536867 0 -1.386294 6 0 0.854415
    -0.776529 3.539509 47 -1.386294 0 -1.386294 6 0 1.047319
    0.223144 3.244544 63 -1.386294 0 -1.386294 6 0 1.047319
    0.254642 3.604138 65 -1.386294 0 -1.386294 6 0 1.266948
    -1.347074 3.598681 63 1.266948 0 -1.386294 6 0 1.266948
    1.61343 3.022861 63 -1.386294 0 -0.597837 7 30 1.266948
    1.477049 2.998229 67 -1.386294 0 -1.386294 7 5 1.348073
    1.205971 3.442019 57 -1.386294 0 -0.430783 7 5 1.398717
    1.541159 3.061052 66 -1.386294 0 -1.386294 6 0 1.446919
    -0.415515 3.516013 70 1.244155 0 -0.597837 7 30 1.470176
    2.288486 3.649359 66 -1.386294 0 0.371564 6 0 1.492904
    -0.562119 3.267666 41 -1.386294 0 -1.386294 6 0 1.558145
    0.182322 3.825375 70 1.658228 0 -1.386294 6 0 1.599388
    1.147402 3.419365 59 -1.386294 0 -1.386294 6 0 1.638997
    2.059239 3.501043 60 1.474763 0 1.348073 7 20 1.658228
    -0.544727 3.37588 59 -0.798508 0 -1.386294 6 0 1.695616
    1.781709 3.451574 63 0.438255 0 1.178655 7 60 1.713798
    0.385262 3.6674 69 1.599388 0 -1.386294 6 0 1.731656
    1.446919 3.124565 68 0.300105 0 -1.386294 6 0 1.766442
    0.512824 3.719651 65 -1.386294 0 -0.798508 7 70 1.800058
    -0.400478 3.865979 67 1.816452 0 -1.386294 7 20 1.816452
    1.040277 3.128951 67 0.223144 0 0.04879 7 80 1.848455
    2.409644 3.37588 65 -1.386294 0 1.619388 6 0 1.894617
    0.285179 4.090169 65 1.962908 0 -0.798508 6 0 1.924249
    0.182322 3.804438 65 1.704748 0 -1.386294 6 0 2.008214
    1.275363 3.037354 71 1.266948 0 -1.386294 6 0 2.008214
    0.00995 3.267666 54 -1.386294 0 -1.386294 6 0 2.021548
    -0.01005 3.216874 63 -1.386294 0 -0.798508 6 0 2.047693
    1.308333 4.11985 64 2.171337 0 -1.386294 7 5 2.085672
    1.423108 3.657131 73 -0.579819 0 1.658228 8 15 2.157559
    0.457425 2.374906 64 -1.386294 0 -1.386294 7 15 2.191654
    2.660959 4.085136 68 1.373716 1 1.832581 7 35 2.213754
    0.797507 3.013081 56 0.936093 0 -0.162519 7 5 2.277267
    0.620576 3.141995 60 -1.386294 0 -1.386294 9 80 2.297573
    1.442202 3.68261 68 -1.386294 0 -1.386294 7 10 2.307573
    0.582216 3.865979 62 1.713798 0 -0.430783 6 0 2.327278
    1.771557 3.896909 61 -1.386294 0 0.81093 7 6 2.374906
    1.48614 3.409496 66 1.7492 0 -0.430783 7 20 2.521721
    1.663926 3.392829 61 0.615186 0 -1.386294 7 15 2.553344
    2.727853 3.995445 79 1.879465 1 2.656757 9 100 2.568788
    1.163151 4.035125 68 1.713798 0 -0.430783 7 40 2.568788
    1.745716 3.498022 43 -1.386294 0 -1.386294 6 0 2.591516
    1.22083 3.568123 70 1.373716 0 -0.798508 6 0 2.591516
    1.091923 3.993603 68 -1.386294 0 -1.386294 7 50 2.656757
    1.660131 4.234831 64 2.073172 0 -1.386294 6 0 2.677591
    0.512824 3.633631 64 1.492904 0 0.04879 7 70 2.68444
    2.127041 4.121473 68 1.766442 0 1.446919 7 40 2.691243
    3.15359 3.516013 59 -1.386294 0 -1.386294 7 5 2.704711
    1.266948 4.280132 66 2.122262 0 -1.386294 7 15 2.718001
    0.97456 2.865054 47 -1.386294 0 0.500775 7 4 2.788093
    0.463734 3.764682 49 1.423108 0 -1.386294 6 0 2.794228
    0.542324 4.178226 70 0.438255 0 -1.386294 7 20 2.806386
    1.061257 3.851211 61 1.294727 0 -1.386294 7 40 2.81241
    0.457425 4.524502 73 2.326302 0 -1.386294 6 0 2.841998
    1.997418 3.719651 63 1.619388 1 1.909542 7 40 2.853592
    2.775709 3.524889 72 -1.386294 0 1.558145 9 95 2.853592
    2.034706 3.917011 66 2.008214 1 2.110213 7 60 2.882004
    2.073172 3.623007 64 -1.386294 0 -1.386294 6 0 2.882004
    1.458615 3.836221 61 1.321756 0 -0.430783 7 20 2.88759
    2.022871 3.878466 68 1.783391 0 1.321756 7 70 2.92047
    2.198335 4.050915 72 2.307573 0 -0.430783 7 10 2.962692
    -0.446287 4.408547 69 -1.386294 0 -1.386294 6 0 2.962692
    1.193922 4.780383 72 2.326302 0 -0.798508 7 5 2.972975
    1.86408 3.593194 60 -1.386294 1 1.321756 7 60 3.013081
    1.160021 3.341093 77 1.7492 0 -1.386294 7 25 3.037354
    1.214913 3.825375 69 -1.386294 1 0.223144 7 20 3.056357
    1.838961 3.236716 60 0.438255 1 1.178655 9 90 3.075006
    2.999226 3.849083 69 -1.386294 1 1.909542 7 20 3.275256
    3.14113 3.263849 68 -0.051293 1 2.420368 7 50 3.337547
    2.010895 4.433789 72 2.122262 0 0.500775 7 60 3.392829
    2.537657 4.354784 78 2.326302 0 -1.386294 7 10 3.435599
    2.6483 3.582129 69 -1.386294 1 2.583998 7 70 3.457893
    2.77944 3.823192 63 -1.386294 0 0.371564 7 50 3.513037
    1.467874 3.070376 66 0.559616 0 0.223144 7 40 3.516013
    2.513656 3.473518 57 0.438255 0 2.327278 7 60 3.530763
    2.613007 3.888754 77 -0.527633 1 0.559616 7 30 3.565298
    2.677591 3.838376 65 1.115142 0 1.7492 9 70 3.57094
    1.562346 3.709907 60 1.695616 0 0.81093 7 30 3.587677
    3.302849 3.51898 64 -1.386294 1 2.327278 7 60 3.630985
    2.024193 3.731699 58 1.638997 0 -1.386294 6 0 3.680091
    1.731656 3.369018 62 -1.386294 1 0.300105 7 30 3.712352
    2.807594 4.718052 65 -1.386294 1 2.463853 7 60 3.984344
    1.562346 3.69511 76 0.936093 1 0.81093 7 75 3.993603
    3.246491 4.101817 68 -1.386294 0 -1.386294 6 0 4.029806
    2.532903 3.677566 61 1.348073 1 -1.386294 7 15 4.129551
    2.830268 3.876396 68 -1.386294 1 1.321756 7 60 4.385147
    3.821004 3.896909 44 -1.386294 1 2.169054 7 40 4.684443
    2.907447 3.396185 52 -1.386294 1 2.463853 7 10 5.143124
    2.882564 3.77391 68 1.558145 1 1.558145 7 80 5.477509
    3.471966 3.974998 68 0.438255 1 2.904165 7 20 5.582932
]

covariates = prostate[:, 1:8]
standardized = (covariates .- mean(covariates; dims = 1)) ./ std(covariates; dims = 1)
data = (y = prostate[:, 9], X = [ones(size(prostate, 1)) standardized])
nothing #hide

# The covariates are standardized to have zero mean and unit variance, and the first
# column of ``X`` is the intercept.
#
# ## Model
#
# The regression is
#
# ```math
# y = X\beta + \varepsilon, \qquad \varepsilon \sim N(0, \sigma^2 I_n),
# ```
#
# with the hierarchical L2-regularization prior
#
# ```math
# \begin{aligned}
# \beta \mid \sigma^2, \psi^2 &\sim N(0, \sigma^2 \psi^2 I_p), \\
# \sigma^2 &\sim \text{Inv-}\chi^2(\nu_0, \tau_0^2), \\
# \psi^2 &\sim \text{Inv-}\chi^2(\omega_0, \psi_0^2),
# \end{aligned}
# ```
#
# where ``\lambda = 1/\psi^2`` is the regularization parameter of ridge regression. As
# in the lab, the intercept is not regularized but has the noninformative prior
# ``\beta_0 \mid \sigma^2 \sim N(0, \kappa_0^2 \sigma^2)`` with ``\kappa_0 = 100``. In
# what follows ``\beta`` includes the intercept, ``p = 8`` is the number of covariates,
# and the prior of ``\beta`` is ``N(0, \sigma^2 \Omega_0^{-1})`` with
# ``\Omega_0 = \text{diag}(\kappa_0^{-2}, \psi^{-2}, \dots, \psi^{-2})``.
#
# The prior hyperparameters are the ones in the lab.

prior = (ν₀ = 0.01, τ₀² = 1.0, ω₀ = 0.01, ψ₀² = 1.0, κ₀ = 100.0)
nothing #hide

# The scaled inverse chi-squared distribution is an inverse gamma distribution.

InvChisq(ν, τ²) = InverseGamma(ν / 2, ν * τ² / 2)

Ω₀(ψ², prior, p) = Diagonal([1 / prior.κ₀^2; fill(1 / ψ², p)])
nothing #hide

# ## Gibbs sampler
#
# The sampler in Box 16.2 has two blocks, ``(\beta, \sigma^2)`` and ``\psi^2``:
#
# ```math
# \begin{aligned}
# \text{Block 1:} \quad
# \beta \mid \sigma^2, \psi^2, y &\sim N\big(\hat\beta_{L_2}, \sigma^2 \Omega_n^{-1}\big), \\
# \sigma^2 \mid \psi^2, y &\sim \text{Inv-}\chi^2(\nu_n, \tau_n^2), \\
# \text{Block 2:} \quad
# \psi^2 \mid \beta, \sigma^2, y &\sim \text{Inv-}\chi^2(\omega_n, \psi_n^2),
# \end{aligned}
# ```
#
# where ``\hat\beta_{L_2} = \Omega_n^{-1} X^\top y`` is the ridge estimator and, from Box
# 5.3 with prior mean zero,
#
# ```math
# \Omega_n = X^\top X + \Omega_0, \qquad \nu_n = \nu_0 + n, \qquad
# \nu_n \tau_n^2 = \nu_0 \tau_0^2 + y^\top y - \hat\beta_{L_2}^\top \Omega_n \hat\beta_{L_2},
# ```
#
# and
#
# ```math
# \omega_n = \omega_0 + p, \qquad
# \psi_n^2 = \Big(\sum_{j=1}^p (\beta_j/\sigma)^2 + \omega_0 \psi_0^2\Big) \Big/ \omega_n .
# ```
#
# The sum in ``\psi_n^2`` is over the regularized coefficients, so it excludes the
# intercept.

function conjugate(ψ², data, prior)
    (; y, X) = data
    Ωₙ = Symmetric(X'X + Ω₀(ψ², prior, size(X, 2) - 1))
    β̂ = Ωₙ \ (X'y)
    νₙ = prior.ν₀ + length(y)
    τₙ² = (prior.ν₀ * prior.τ₀² + y'y - β̂' * Ωₙ * β̂) / νₙ
    return (; Ωₙ, β̂, νₙ, τₙ²)
end

function σ²_conditional(state, data, prior)
    (; νₙ, τₙ²) = conjugate(state.ψ², data, prior)
    return InvChisq(νₙ, τₙ²)
end

function β_conditional(state, data, prior)
    (; Ωₙ, β̂) = conjugate(state.ψ², data, prior)
    return MvNormal(β̂, Symmetric(state.σ² * inv(Ωₙ)))
end

function ψ²_conditional(state, data, prior)
    slopes = state.β[2:end]
    ωₙ = prior.ω₀ + length(slopes)
    return InvChisq(ωₙ, (sum(abs2, slopes) / state.σ² + prior.ω₀ * prior.ψ₀²) / ωₙ)
end
nothing #hide

# Block 1 is drawn in two steps, first ``\sigma^2`` and then ``\beta`` given
# ``\sigma^2``. The model is therefore set up with the three conditionals in the order
# ``\sigma^2``, ``\beta``, ``\psi^2``, and each sweep is exactly the two-block sampler
# of Box 16.2. Note that the conditional of ``\sigma^2`` does not depend on ``\beta``.

loglik(θ, data) = logpdf(MvNormal(data.X * θ.β, θ.σ² * I), data.y)

function logprior(θ, prior)
    p = length(θ.β) - 1
    return logpdf(MvNormalCanon(Matrix(Ω₀(θ.ψ², prior, p) / θ.σ²)), θ.β) +
        logpdf(InvChisq(prior.ν₀, prior.τ₀²), θ.σ²) + logpdf(InvChisq(prior.ω₀, prior.ψ₀²), θ.ψ²)
end

ridge = GibbsModel(
    conditionals = (σ² = σ²_conditional, β = β_conditional, ψ² = ψ²_conditional); loglik, logprior)
nothing #hide

# ## Posterior
#
# A posterior sample of 10000 draws after a burn-in of 1000 draws, as in the lab.

rng = Xoshiro(2024)
init = (σ² = 1.0, β = zeros(9), ψ² = 1.0)
draws = gibbs(rng, ridge, data, prior, init; ndraws = 10_000, burnin = 1_000)

names = ["intercept", "lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45"]
summary(name, x) = println(rpad(name, 10), "mean = ", lpad(round(mean(x); digits = 3), 6),
    "   std = ", round(std(x); digits = 3))
for (j, name) in enumerate(names)
    summary(name, [draw.β[j] for draw in draws])
end
summary("σ²", [draw.σ² for draw in draws])
summary("ψ²", [draw.ψ² for draw in draws])
summary("λ", [1 / draw.ψ² for draw in draws])

# ## Marginal likelihood
#
# With the blocks in this order, the posterior ordinate is decomposed as
#
# ```math
# \pi(\theta^* \mid y) = \pi(\sigma^{2*} \mid y)\, \pi(\beta^* \mid y, \sigma^{2*})\,
#   \pi(\psi^{2*} \mid y, \sigma^{2*}, \beta^*).
# ```
#
# The first factor averages the ``\text{Inv-}\chi^2(\nu_n, \tau_n^2)`` density over the
# draws of ``\psi^2`` from the full sampler. The second averages the normal density of
# ``\beta`` over the draws of ``\psi^2`` from a reduced run with ``\sigma^2`` fixed. The
# third is the full conditional of ``\psi^2``, which is known exactly.

estimate = chib(rng, ridge, data, prior, init; ndraws = 10_000, burnin = 1_000)

# The log ordinates of the three blocks are

estimate.logordinates

# ## Check by numerical integration
#
# Given ``\psi^2`` the prior is conjugate, so the marginal likelihood conditional on
# ``\psi^2`` is known in closed form,
#
# ```math
# m(y \mid \psi^2) = \pi^{-n/2} \frac{|\Omega_0|^{1/2}}{|\Omega_n|^{1/2}}
#   \frac{\Gamma(\nu_n/2)}{\Gamma(\nu_0/2)}
#   \frac{(\nu_0 \tau_0^2)^{\nu_0/2}}{(\nu_n \tau_n^2)^{\nu_n/2}} .
# ```
#
# The marginal likelihood is then a one-dimensional integral over the prior of
# ``\psi^2``,
#
# ```math
# m(y) = \int m(y \mid \psi^2)\, p(\psi^2)\, d\psi^2,
# ```
#
# which is computed here by the trapezoidal rule on a grid for ``\ln \psi^2``.

function logmarglik(ψ², data, prior)
    (; Ωₙ, νₙ, τₙ²) = conjugate(ψ², data, prior)
    n, ν₀ = length(data.y), prior.ν₀
    return -n / 2 * log(π) + (logdet(Ω₀(ψ², prior, size(data.X, 2) - 1)) - logdet(Ωₙ)) / 2 +
        loggamma(νₙ / 2) - loggamma(ν₀ / 2) + ν₀ / 2 * log(ν₀ * prior.τ₀²) - νₙ / 2 * log(νₙ * τₙ²)
end

grid = range(-40, 40; length = 8001)   # ln ψ²
logintegrand = [logmarglik(exp(u), data, prior) + logpdf(InvChisq(prior.ω₀, prior.ψ₀²), exp(u)) + u
                for u in grid]
integrand = exp.(logintegrand .- maximum(logintegrand))
exact = log(step(grid) * (sum(integrand) - (integrand[1] + integrand[end]) / 2)) + maximum(logintegrand)

# The estimate is within a few numerical standard errors of the exact value:

(estimate = estimate.logmarglik, exact = exact, nse = estimate.nse)
