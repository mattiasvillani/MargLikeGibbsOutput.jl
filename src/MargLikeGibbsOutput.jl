"""
    MargLikeGibbsOutput

Marginal likelihood from the Gibbs output, following

Chib, S. (1995). Marginal Likelihood from the Gibbs Output.
*Journal of the American Statistical Association*, 90(432), 1313-1321.
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
