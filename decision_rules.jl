@doc raw"""Decision Rules

Decision rules correspond to some user-defined function in terms of the predictive mean, predictive standard deviation,
some hyperparameters θ, and functional access to an object sx. Here, sx is the evaluation of our surrogate at some
arbitrary location x and the object has some useful data cached onto it.

To construct a decision rule, one merely defines a function, as defined above, and pass it to the DecisionRule
constructor and provide it with a name, i.e.:
```
g(μ, σ, θ, sx) = μ - θ[1]*σ + exp(minimum(get_observations(sx)))
my_decision = DecisionRule(g, "Just An Example")
```

Once the decision rule is constructed, we employ automatic differentiation to construct a function handle on all of the
necessary derivatives for our computations. We consider all mixed partials up to order 2 with respect to
μ, σ, and θ.
"""
struct DecisionRule{F <: Function} <: AbstractDecisionRule
    g::F        # The function g(μ, σ, θ)
    dg_dμ       # Gradient of g with respect to μ
    d2g_dμ      # Hessian of g with respect to μ
    dg_dσ       # Gradient of g with respect to σ
    d2g_dσ      # Hessian of g with respect to σ
    dg_dθ       # Gradient of g with respect to θ
    d2g_dθ      # Hessian of g with respect to θ
    d2g_dμdθ    # Mixed derivative of g with respect to μ and θ
    d2g_dσdθ    # Mixed derivative of g with respect to σ and θ
    name::String
end

function Base.show(io::IO, r::DecisionRule{F}) where F
    print(io, "DecisionRule{$(r.name)}")
end

get_name(dr::DecisionRule) = dr.name

function DecisionRule(g::Function, name::String)
    dg_dμ(μ, σ, θ, sx) = ForwardDiff.derivative(μ -> g(μ, σ, θ, sx), μ)
    d2g_dμ(μ, σ, θ, sx) = ForwardDiff.derivative(μ -> dg_dμ(μ, σ, θ, sx), μ)
    dg_dσ(μ, σ, θ, sx) = ForwardDiff.derivative(σ -> g(μ, σ, θ, sx), σ)
    d2g_dσ(μ, σ, θ, sx) = ForwardDiff.derivative(σ -> dg_dσ(μ, σ, θ, sx), σ)
    dg_dθ(μ, σ, θ, sx) = ForwardDiff.gradient(θ -> g(μ, σ, θ, sx), θ)
    d2g_dθ(μ, σ, θ, sx) = ForwardDiff.hessian(θ -> g(μ, σ, θ, sx), θ)
    d2g_dμdθ(μ, σ, θ, sx) = ForwardDiff.gradient(θ -> dg_dμ(μ, σ, θ, sx), θ)
    d2g_dσdθ(μ, σ, θ, sx) = ForwardDiff.gradient(θ -> dg_dσ(μ, σ, θ, sx), θ)

    return DecisionRule(g, dg_dμ, d2g_dμ, dg_dσ, d2g_dσ, dg_dθ, d2g_dθ, d2g_dμdθ, d2g_dσdθ, name)
end

(bp::DecisionRule)(μ::Number, σ::Number, θ::AbstractVector, sx) = bp.g(μ, σ, θ, sx)


function first_partial(p::AbstractDecisionRule; symbol::Symbol)
    if symbol == :μ
        return p.dg_dμ
    elseif symbol == :σ
        return p.dg_dσ
    elseif symbol == :θ
        return p.dg_dθ
    else
        error("Unknown symbol. Use :μ, :σ, or :θ")
    end
end
first_partials(p::AbstractDecisionRule) = (
    μ=first_partial(p, symbol=:μ),
    σ=first_partial(p, symbol=:σ),
    θ=first_partial(p, symbol=:θ)
)

function second_partial(p::AbstractDecisionRule; symbol::Symbol)
    if symbol == :μ
        return p.d2g_dμ
    elseif symbol == :σ
        return p.d2g_dσ
    elseif symbol == :θ
        return p.d2g_dθ
    else
        error("Unknown symbol. Use :μ, :σ, or :θ")
    end
end
second_partials(p::AbstractDecisionRule) = (
    μ=second_partial(p, symbol=:μ),
    σ=second_partial(p, symbol=:σ),
    θ=second_partial(p, symbol=:θ)
)

function mixed_partial(p::AbstractDecisionRule; symbol::Symbol)
    if symbol == :μθ
        p.d2g_dμdθ 
    elseif symbol == :σθ
        p.d2g_dσdθ
    else
        error("Unknown symbol. Use :μθ or :σθ")
    end
end

# Some Common Acquisition Functions
function EI(; σtol=1e-8)
    # TODO: Create a lazy evaluation version of this function. I want to return ei(μ, σ, θ, minimum(y))
    function ei(μ, σ, θ, sx)
        if σ < σtol
            return 0.
        end
        fmini = minimum(get_observations(sx))
        improvement = fmini - μ - θ[1]
        z = improvement / σ
        
        expected_improvement = improvement*Distributions.normcdf(z) + σ*Distributions.normpdf(z)
        return expected_improvement
    end

    return DecisionRule(ei, "EI")
end

function POI(; σtol=1e-8)
    function poi(μ, σ, θ, sx)
        if σ < σtol
            return 0.0
        end
        fmini = minimum(get_observations(sx))
        improvement = fmini - μ - θ[1]
        z = improvement / σ

        probability_improvement = Distributions.normcdf(z)
        return probability_improvement
    end

    return DecisionRule(poi, "POI")
end

function LCB()
    @doc raw"""Lower confidence bound is μ(x) - κ*σ(x), a quantity we want to minimize in our minimization
    framework for Bayesian optimization. However, we expect our acquisition functions to be things we want
    to maximize. Hence, minimizing LCB(x) is equivalent to maximizing -LCB(x) = κ*σ(x) - μ(x)
    """
    function lcb(μ, σ, θ, sx)
        return θ[1] * σ - μ
    end

    return DecisionRule(lcb, "LCB")
end

function RandomAcquisition()
    function random_decision(μ, σ, θ, sx)
        return 0.
    end

    return DecisionRule(random_decision, RANDOM_ACQUISITION)
end


# Custom string method for BasePolicy
Base.string(bp::AbstractDecisionRule) = bp.name