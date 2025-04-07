@doc raw"""Decision Rule

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

@doc raw"""
To construct a decision rule, one merely defines a function, as defined above, and pass it to the DecisionRule
constructor and provide it with a name, i.e.:
```
g(μ, σ, θ, sx) = μ - θ[1]*σ
my_decision = DecisionRule(g, "Lower Confidence Bound")
```
"""
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

(dr::DecisionRule)(μ::Number, σ::Number, θ::AbstractVector, sx) = dr.g(μ, σ, θ, sx)


function first_partial(p::DecisionRule; symbol::Symbol)
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
first_partials(p::DecisionRule) = (
    μ=first_partial(p, symbol=:μ),
    σ=first_partial(p, symbol=:σ),
    θ=first_partial(p, symbol=:θ)
)

function second_partial(p::DecisionRule; symbol::Symbol)
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
second_partials(p::DecisionRule) = (
    μ=second_partial(p, symbol=:μ),
    σ=second_partial(p, symbol=:σ),
    θ=second_partial(p, symbol=:θ)
)

function mixed_partial(p::DecisionRule; symbol::Symbol)
    if symbol == :μθ
        p.d2g_dμdθ 
    elseif symbol == :σθ
        p.d2g_dσdθ
    else
        error("Unknown symbol. Use :μθ or :σθ")
    end
end

# Some Common Acquisition Functions
@doc raw"""
    EI(; σtol=1e-8)

Expected Improvement (EI) is an acquisition function used in Bayesian optimization. It quantifies the 
expected amount of improvement over the current best observed value, taking into account both the 
predicted mean and uncertainty of the surrogate model.

# Definition
The EI is computed as:

    EI(x) = E[max(f_min - f(x) - ξ, 0)],

where:
- `f(x)` is the objective value predicted by the surrogate model at \( x \),
- `f_min` is the current minimum observed value,
- `ξ` (theta[1]) is a small positive margin (exploration parameter),
- `E` is the expectation operator.

To calculate EI, the improvement is defined as:

    improvement = f_min - μ(x) - ξ,

where:
- `μ(x)` is the posterior mean, and
- `σ(x)` is the posterior standard deviation.

The standard normal score \( z \) is then:

    z = improvement / σ(x).

The EI is given by:

    EI(x) = improvement * Φ(z) + σ(x) * φ(z),

where:
- \( Φ(z) \) is the CDF of the standard normal distribution,
- \( φ(z) \) is the PDF of the standard normal distribution.

# Arguments
- `σtol=1e-8`: A small threshold for the standard deviation. If \( σ(x) < σtol \), the EI is set to 0.0 
  to avoid numerical instability.

# Returns
A `DecisionRule` object representing the Expected Improvement acquisition function.

# Notes
- EI balances exploration and exploitation by favoring regions with high predicted improvement or high 
  uncertainty.
- If the standard deviation \( σ(x) \) is very small, the EI is effectively zero, as no significant 
  improvement is expected.
"""
function EI(σtol=1e-8)
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

@doc raw"""
    POI(; σtol=1e-8)

Probability of Improvement (POI) is an acquisition function used in Bayesian optimization. It measures 
the probability that the current model will produce an improvement over the best observed value.

# Definition
The POI is computed as:

    POI(x) = P(f(x) < f_min - ξ),

where:
- `f(x)` is the objective value predicted by the surrogate model at \( x \),
- `f_min` is the current minimum observed value,
- `ξ` (theta[1]) is a small positive margin (exploration parameter),
- `P` is the cumulative distribution function (CDF) of a standard normal distribution.

To compute the POI, the improvement is defined as:

    improvement = f_min - μ(x) - ξ,

where:
- `μ(x)` is the posterior mean, and
- `σ(x)` is the posterior standard deviation.

The standard normal score \( z \) is then:

    z = improvement / σ(x).

The POI is finally given by:

    POI(x) = Φ(z),

where \( Φ \) is the CDF of the standard normal distribution.

# Arguments
- `σtol=1e-8`: A small threshold for the standard deviation. If \( σ(x) < σtol \), the POI is set to 0.0 
  to avoid numerical instability.

# Returns
A `DecisionRule` object representing the Probability of Improvement acquisition function.

# Notes
- If the standard deviation \( σ(x) \) is too small (indicating high confidence), no exploration is 
  encouraged, and the POI is zero.
"""
function POI(σtol=1e-8)
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

@doc raw"""
    LCB()

Lower Confidence Bound (LCB) is a commonly used acquisition function in Bayesian optimization. 
In its original form, it is defined as:

    LCB(x) = μ(x) - κ * σ(x),

where:
- `μ(x)` is the posterior mean,
- `σ(x)` is the posterior standard deviation,
- `κ` is a positive scaling parameter that controls the exploration-exploitation trade-off.

In a minimization framework, we aim to minimize LCB(x). However, since acquisition functions 
in this framework are expected to be maximized, we instead maximize:

    -LCB(x) = κ * σ(x) - μ(x).

This transformation allows the minimization task to fit seamlessly into the framework's maximization 
structure.

# Returns
A `DecisionRule` object representing the transformed LCB acquisition function.
"""
function LCB()
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

decision_rule_mapping = Dict(
    "ei" => EI,
    "poi" => POI,
    "lcb" => LCB
)