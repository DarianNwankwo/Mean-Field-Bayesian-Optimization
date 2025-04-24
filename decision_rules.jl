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
    x::Vector{T},
    cache::SurrogateEvaluationCache;
    atol=1e-10
) where T
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


mutable struct ExpectedImprovement <: AbstractDecisionRule
    minimum::Float64
    exploration::Float64
end
ExpectedImprovement(; mini=Inf, exploration=0.) = ExpectedImprovement(mini, exploration)

function setparams!(dr::ExpectedImprovement, surrogate::AbstractSurrogate)
    dr.minimum = min(
        minimum(get_active_observations(surrogate)),
        dr.minimum
    )
end
get_name(::ExpectedImprovement) = EXPECTED_IMPROVEMENT_NAME

@inline function eval(
    s::AbstractSurrogate,
    ei::ExpectedImprovement,
    x::Vector{T},
    cache::SurrogateEvaluationCache
    ) where T <: Real
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
    x::Vector{T},
    cache::SurrogateEvaluationCache
    ) where T <: Real
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


mutable struct ProbabilityOfImprovement <: AbstractDecisionRule
    minimum::Float64
end
ProbabilityOfImprovement(; mini=Inf) = ProbabilityOfImprovement(mini)

function setparams!(poi::ProbabilityOfImprovement, surrogate::AbstractSurrogate)
    poi.minimum = min(
        minimum(get_active_observations(surrogate)),
        poi.minimum
    )
end
get_name(::ProbabilityOfImprovement) = PROBABILITY_OF_IMPROVEMENT_NAME

@inline function eval(
    s::AbstractSurrogate,
    poi::ProbabilityOfImprovement,
    x::Vector{T},
    cache::SurrogateEvaluationCache
    ) where T <: Real
    m = evaluate_moments_and_derivatives!(s, x, cache)
    σ2 = m.σ^2
    σ2 < POI_VARIANCE_TOLERANCE && return float(m.μ < poi.minimum)
    z = (poi.minimum - m.μ) / m.σ
    return Distributions.normcdf(z)
end

@inline function eval_gradient(
    s::AbstractSurrogate,
    poi::ProbabilityOfImprovement,
    x::Vector{T},
    cache::SurrogateEvaluationCache
    ) where T <: Real
    m = evaluate_moments_and_derivatives!(s, x, cache)
    σ2 = m.σ^2
    σ2 < POI_VARIANCE_TOLERANCE && return begin
        m.∇σ .= float(m.μ < poi.minimum)
        invalidate!(cache)
        return m.∇σ
    end
    z = (poi.minimum - m.μ) / m.σ
    return Distributions.normpdf(z)
end


mutable struct RandomSampler <: AbstractDecisionRule end
setparams!(rs::RandomSampler, surrogate::AbstractSurrogate) = nothing
get_name(::RandomSampler) = RANDOM_SAMPLER_NAME

@inline function eval(
    s::AbstractSurrogate,
    rs::RandomSampler,
    x::Vector{T},
    cache::SurrogateEvaluationCache
    ) where T <: Real
    m = evaluate_moments_and_derivatives!(s, x, cache)
    return 0.
end

@inline function eval_gradient(
    s::AbstractSurrogate,
    rs::RandomSampler,
    x::Vector{T},
    cache::SurrogateEvaluationCache
    ) where T <: Real
    m = evaluate_moments_and_derivatives!(s, x, cache)
    return 0. * m.∇μ
end


mutable struct UpperConfidenceBound <: AbstractDecisionRule
    beta::Float64
end
setparams!(ucb::UpperConfidenceBound, surrogate::AbstractSurrogate) = nothing
get_name(::UpperConfidenceBound) = UPPER_CONFIDENCE_BOUND_NAME

@inline function eval(
    s::AbstractSurrogate,
    ucb::UpperConfidenceBound,
    x::Vector{T},
    cache::SurrogateEvaluationCache
    ) where T <: Real
    m = evaluate_moments_and_derivatives!(s, x, cache)
    return m.μ + ucb.beta * m.σ
end

@inline function eval_gradient(
    s::AbstractSurrogate,
    ucb::UpperConfidenceBound,
    x::Vector{T},
    cache::SurrogateEvaluationCache
    ) where T <: Real
    m = evaluate_moments_and_derivatives!(s, x, cache)
    return m.∇μ + ucb.beta * m.∇σ
end