mutable struct SurrogateEvaluationCache
    x::Vector{Float64}          # Last input used for evaluation
    μ::Float64                  # Cached mean
    σ::Float64                  # Cached standard deviation
    ∇μ::Vector{Float64}         # Cached gradient of the mean
    ∇σ::Vector{Float64}         # Cached gradient of the standard deviation
    valid::Bool                 # Flag indicating if the cache is valid
end

# Constructor for the cache, assuming dimension `d` for the input x.
function SurrogateEvaluationCache(d::Int)
    return SurrogateEvaluationCache(
        zeros(d),
        0.0,
        0.0,
        zeros(d),
        zeros(d),
        false
    )
end
invalidate!(cache::SurrogateEvaluationCache) = cache.valid = false

function evaluate_moments_and_derivatives!(
    s::AbstractSurrogate,
    x::Vector{Float64},
    cache::SurrogateEvaluationCache;
    atol=1e-10
)
    if cache.valid && all(abs.(x .- cache.x) .< atol)
        return (; μ=cache.μ, σ=cache.σ, ∇μ=cache.∇μ, ∇σ=cache.∇σ)
    end

    # If not, compute and update the cache.
    μ, σ = compute_moments(s, x)
    ∇μ, ∇σ = compute_moments_gradient(s, x)

    copyto!(cache.x, x)
    cache.μ = μ
    cache.σ = σ
    copyto!(cache.∇μ, ∇μ)
    copyto!(cache.∇σ, ∇σ)
    cache.valid = true

    return (; μ, σ, ∇μ, ∇σ)
end

struct ExpectedImprovement <: AbstractDecisionRule
    minimum::Float64
    exploration::Float64
end
set_minimum!(dr::ExpectedImprovement, mm::Float64) = dr.minimum = mm

@inline function eval(
    s::AbstractSurrogate,
    ei::ExpectedImprovement,
    x::Vector{Float64},
    cache::SurrogateEvaluationCache
)
    m = evaluate_moments_and_derivatives!(s, x, cache)
    σ2 = m.σ^2
    σ2 < EI_VARIANCE_TOLERANCE && return 0.
    μbar = (ei.minimum - m.μ - ei.exploration)
    z = μbar / m.σ
    return μbar * Distributions.normcdf(z) + m.σ * Distributions.normpdf(z)
end


@inline function eval_gradient(
    s::AbstractSurrogate,
    ei::ExpectedImprovement,
    x::Vector{Float64},
    cache::SurrogateEvaluationCache
)
    # Use the persistent cache to avoid recomputation.
    m = evaluate_moments_and_derivatives!(s, x, cache)
    σ2 = m.σ^2
    if σ2 < EI_VARIANCE_TOLERANCE
        return 0. * m.∇μ
    end

    μbar = (ei.minimum - m.μ - ei.exploration)
    z = μbar / m.σ
    ∇z = (-m.∇μ - z * m.∇σ) / m.σ
    Φz = Distributions.normcdf(z)
    g = z * Φz + Distributions.normpdf(z)
    
    return g * m.∇σ + m.σ * Φz * ∇z
end