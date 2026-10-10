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
