using MargLikeGibbsOutput
using Distributions, LinearAlgebra, Random, Test

@testset "MargLikeGibbsOutput" begin

    @testset "one block: the ordinate is exact" begin
        # yᵢ ~ N(μ, 1) with μ ~ N(0, 1/κ)
        rng = Xoshiro(1)
        y, κ = randn(rng, 20) .+ 1, 0.5
        model = GibbsModel(
            conditionals = (μ = (state, y, κ) -> Normal(sum(y) / (length(y) + κ), 1 / sqrt(length(y) + κ)),),
            loglik = (θ, y) -> sum(logpdf.(Normal(θ.μ, 1), y)),
            logprior = (θ, κ) -> logpdf(Normal(0, 1 / sqrt(κ)), θ.μ),
        )
        estimate = chib(rng, model, y, κ, (μ = 0.0,); ndraws = 100, burnin = 10)
        @test estimate.logmarglik ≈ logpdf(MvNormal(zeros(20), I + fill(1 / κ, 20, 20)), y)
        @test estimate.nse < 1e-12
    end

    @testset "three blocks: conjugate regression" begin
        # y ~ N(x₁β₁ + x₂β₂, σ²I) with βⱼ|σ² ~ N(0, σ²/κ) and σ² ~ IG(a, b)
        rng = Xoshiro(2)
        n = 30
        x₁, x₂ = randn(rng, n), randn(rng, n)
        data = (; x₁, x₂, y = x₁ - 0.5x₂ + randn(rng, n))
        prior = (κ = 2.0, a = 3.0, b = 2.0)

        β_conditional(x, residual, σ², κ) = Normal(dot(x, residual) / (dot(x, x) + κ), sqrt(σ² / (dot(x, x) + κ)))
        residuals(θ, data) = data.y - data.x₁ * θ.β₁ - data.x₂ * θ.β₂
        model = GibbsModel(
            conditionals = (
                β₁ = (s, data, prior) -> β_conditional(data.x₁, data.y - data.x₂ * s.β₂, s.σ², prior.κ),
                β₂ = (s, data, prior) -> β_conditional(data.x₂, data.y - data.x₁ * s.β₁, s.σ², prior.κ),
                σ² = (s, data, prior) -> InverseGamma(prior.a + (n + 2) / 2,
                    prior.b + (sum(abs2, residuals(s, data)) + prior.κ * (s.β₁^2 + s.β₂^2)) / 2),
            ),
            loglik = (θ, data) -> logpdf(MvNormal(zeros(n), θ.σ² * I), residuals(θ, data)),
            logprior = (θ, prior) -> sum(logpdf.(Normal(0, sqrt(θ.σ² / prior.κ)), (θ.β₁, θ.β₂))) +
                logpdf(InverseGamma(prior.a, prior.b), θ.σ²),
        )
        X = [x₁ x₂]
        exact = logpdf(MvTDist(2prior.a, zeros(n), Matrix(prior.b / prior.a * (I + X * X' / prior.κ))), data.y)

        init = (β₁ = 0.0, β₂ = 0.0, σ² = 1.0)
        estimate = chib(rng, model, data, prior, init)
        @test 0 < estimate.nse < 0.05
        @test abs(estimate.logmarglik - exact) < 4 * estimate.nse
        @test sum(estimate.logordinates) ≈ estimate.loglik + estimate.logprior - estimate.logmarglik
        @test keys(estimate.θstar) == (:β₁, :β₂, :σ²)

        # The identity holds at any point, given as a function of the draws or directly
        atmode = chib(rng, model, data, prior, init; θstar = posteriormode)
        @test abs(atmode.logmarglik - exact) < 4 * atmode.nse
        θstar = (β₁ = 0.8, β₂ = -0.3, σ² = 1.2)
        fixed = chib(rng, model, data, prior, init; θstar)
        @test fixed.θstar == θstar
        @test abs(fixed.logmarglik - exact) < 4 * fixed.nse

        draws = gibbs(rng, model, data, prior, init; ndraws = 500, burnin = 100)
        @test length(draws) == 500
        @test posteriormean(draws, nothing).β₁ ≈ sum(d.β₁ for d in draws) / 500
    end

    @testset "latent data: probit regression of Chib (1995), Table 2" begin
        include(joinpath(@__DIR__, "..", "examples", "nodal.jl"))
        @test all(abs(estimate.logmarglik - chib1995) < 0.06 for (estimate, chib1995) in results)
        @test all(0 < estimate.nse < 0.03 for (estimate, _) in results)
    end
end
