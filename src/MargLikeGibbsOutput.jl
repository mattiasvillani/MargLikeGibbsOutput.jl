"""
    MargLikeGibbsOutput

Estimation of the marginal likelihood from the output of a Gibbs sampler, by the method of
Chib, S. (1995). Marginal Likelihood from the Gibbs Output. *Journal of the American
Statistical Association*, 90(432), 1313-1321.

The package works for any number of Gibbs blocks, with or without latent data, as long as
the full conditional posterior of every parameter block has a known normalizing constant.
Specify the model with [`GibbsModel`](@ref), simulate the posterior with [`gibbs`](@ref)
and estimate the log marginal likelihood with [`chib`](@ref).
"""
module MargLikeGibbsOutput

using Distributions: logpdf
using LinearAlgebra: dot
using LogExpFunctions: logsumexp
using Random: AbstractRNG, default_rng
using Statistics: mean

include("gibbs.jl")
export GibbsModel, gibbs

include("chib.jl")
export chib, ChibEstimate, posteriormean, posteriormode

end
