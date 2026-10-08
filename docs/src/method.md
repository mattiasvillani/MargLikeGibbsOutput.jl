# Method

This page summarizes the estimator in Chib (1995) and how the package computes it.

## The marginal likelihood identity

Let ``f(y \mid \theta)`` be the likelihood and ``\pi(\theta)`` the prior density. The
marginal likelihood ``m(y)`` is the normalizing constant of the posterior density, so

```math
m(y) = \frac{f(y \mid \theta)\, \pi(\theta)}{\pi(\theta \mid y)}
```

for any ``\theta``. The likelihood and the prior are easy to evaluate at a given point
``\theta^*``, which leaves the posterior ordinate ``\pi(\theta^* \mid y)``. With an
estimate ``\hat\pi(\theta^* \mid y)`` of it, the estimate of the marginal likelihood is

```math
\ln \hat m(y) = \ln f(y \mid \theta^*) + \ln \pi(\theta^*) - \ln \hat\pi(\theta^* \mid y).
```

## The posterior ordinate

Suppose the Gibbs sampler has the parameter blocks ``\theta_1, \dots, \theta_B`` and
possibly latent data ``z``. The posterior ordinate factors as

```math
\pi(\theta^* \mid y) = \prod_{r=1}^B \pi(\theta_r^* \mid y, \theta_1^*, \dots, \theta_{r-1}^*),
```

and each factor is an integral of the full conditional density of block ``r`` over the
blocks that come after it and the latent data,

```math
\pi(\theta_r^* \mid y, \theta_{<r}^*) =
\int \pi(\theta_r^* \mid y, \theta_{<r}^*, \theta_{>r}, z)\,
     \pi(\theta_{>r}, z \mid y, \theta_{<r}^*)\, d\theta_{>r}\, dz.
```

The first density in the integrand is known in closed form, since it is one of the full
conditionals of the Gibbs sampler. The second is the distribution of a *reduced* Gibbs
run: the same sampler, but with the blocks ``\theta_1, \dots, \theta_{r-1}`` held fixed
at ``\theta^*``. The factor is therefore estimated by the average

```math
\hat\pi(\theta_r^* \mid y, \theta_{<r}^*) =
\frac{1}{G} \sum_{g=1}^G \pi\big(\theta_r^* \mid y, \theta_{<r}^*, \theta_{>r}^{(g)}, z^{(g)}\big)
```

over the draws from that run. For ``r = 1`` no block is fixed, and the run is the
ordinary Gibbs sampler.

This is why the full conditionals are all that is needed. In the package, a reduced run
is a call to the same [`gibbs`](@ref) function with the full conditionals of the fixed
blocks left out, and a full conditional that returns a distribution serves both to draw
the block and to evaluate its density at ``\theta_r^*``.

[`chib`](@ref) makes the following runs, each with `ndraws` draws after `burnin`:

1. A run of the full sampler to choose ``\theta^*``. It is skipped if ``\theta^*`` is given.
2. One run for each block ``r = 1, \dots, B``, with the blocks before ``r`` fixed.

The run for the last block is skipped in models without latent data. There is then
nothing left to integrate over, and the full conditional of ``\theta_B`` is the exact
ordinate.

!!! note
    The paper uses the draws from the first run both to choose ``\theta^*`` and to
    estimate the first ordinate, which requires storing the draws of the latent data.
    The package makes a separate run for the first ordinate instead, and stores only the
    parameter blocks.

## Latent data

Latent data are never integrated out analytically. They are redrawn in every run,
conditional on the blocks that are fixed in that run, and the average above integrates
over them. The draws of ``z`` from one run cannot be reused in another, because they
come from different distributions: ``z \mid y`` in the first run but
``z \mid y, \theta_1^*`` in the second.

The likelihood ``f(y \mid \theta^*)`` in the identity must not depend on the latent
data, so the log-likelihood that you supply is the one with ``z`` integrated out.

## Numerical standard error

Let ``h^{(g)}`` be the vector with the ``B`` full conditional densities at draw ``g``,
so that the estimated ordinates are the elements of
``\hat h = G^{-1} \sum_g h^{(g)}``. The variance of ``\hat h`` is estimated by the
Newey-West estimator

```math
\widehat{\text{var}}(\hat h) = \frac{1}{G} \left[ \Omega_0 +
  \sum_{s=1}^q \left(1 - \frac{s}{q+1}\right) (\Omega_s + \Omega_s') \right],
\qquad
\Omega_s = \frac{1}{G} \sum_{g=s+1}^G (h^{(g)} - \hat h)(h^{(g-s)} - \hat h)',
```

where the number of lags ``q`` is the keyword `lags`, which is 10 by default as in the
paper. The log posterior ordinate is ``\sum_r \ln \hat h_r``, so by the delta method the
numerical standard error of ``\ln \hat m(y)`` is

```math
\sqrt{a'\, \widehat{\text{var}}(\hat h)\, a}, \qquad a = (1/\hat h_1, \dots, 1/\hat h_B)'.
```

The standard error measures the variation of the estimate over repeated simulations
with ``\theta^*`` fixed.

## When the sampler does not mix

The estimator assumes that the Gibbs sampler explores the whole posterior. If the
posterior has several modes and the sampler visits only some of them, the posterior
ordinate is overestimated and the marginal likelihood underestimated, and the numerical
standard error gives no warning. The standard case is label switching in mixture
models, where the posterior has ``d!`` identical modes; see Neal (1999) and the examples
on [Finite mixture models](@ref) and the [Markov switching model](@ref).

## References

- Chib, S. (1995). Marginal Likelihood from the Gibbs Output. *Journal of the American
  Statistical Association*, 90(432), 1313-1321.
- Neal, R. M. (1999). Erroneous Results in "Marginal Likelihood from the Gibbs Output".
  Unpublished note, University of Toronto.
