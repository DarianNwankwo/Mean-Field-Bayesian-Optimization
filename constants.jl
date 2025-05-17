const DOUBLE = 2
const JITTER = 1e-6
const NEWTON_SOLVE_TIME_LIMIT = .1
const X_RELTOL = 1e-4
const F_RELTOL = 1e-4
const MAX_DIMENSION = 20
const EI_VARIANCE_TOLERANCE = 1e-8
const CACHE_SAME_X_TOLERANCE = 1e-6
const EXPECTED_IMPROVEMENT_NAME = "EI"
const POI_VARIANCE_TOLERANCE = 1e-8
const PROBABILITY_OF_IMPROVEMENT_NAME = "POI"
const UCB_VARIANCE_TOLERANCE = 1e-8
const UPPER_CONFIDENCE_BOUND_NAME = "UCB"
const RANDOM_SAMPLER_NAME = "Random"

"""
The amount of space to allocate for our surrogate model in terms of the number of 
observations.
"""
const DEFAULT_CAPACITY = 100


"""
Generates linear mappings up to dimension `dim` as named functions to be provided
to our PolynomialBasisFunction struct to reduce the number of times dynamic dispatch
is invoked on anonymous functions.
"""
macro generate_linear_phi(dim)
    # Evaluate the argument in the current module to get its value.
    dim_val = Base.eval(@__MODULE__, dim)
    for j in 1:dim_val
        fname = Symbol("linear_phi_", j)
        @eval $(fname)(x::AbstractVector{T}) where T = x[$j]
    end
    return nothing
end

macro generate_∇linear_phi(dim)
    # Evaluate the argument in the current module to get its value.
    dim_val = Base.eval(@__MODULE__, dim)
    for j in 1:dim_val
        fname = Symbol("∇linear_phi_", j, "!")
        @eval $(fname)(G::AbstractVector{T}, x::AbstractVector{T}) where T = (fill!(G, 0.0); G[$j] = 1.0; return G)
    end
    return nothing
end


@generate_linear_phi MAX_DIMENSION
@generate_∇linear_phi MAX_DIMENSION

"""
Helper function for grabbing the first `dim` linear mappings to be passed to our
PolynomialBasisFunction struct.
"""
function get_linear_phi_functions(dim::Int)
    # @__MODULE__ refers to the current module where the functions are defined.
    return ntuple(i -> getfield(@__MODULE__, Symbol("linear_phi_", i)), dim)
end

function get_∇linear_phi_functions(dim::Int)
    # Returns an NTuple of the first `dim` gradient functions.
    return ntuple(i -> getfield(@__MODULE__, Symbol("∇linear_phi_", i, "!")), dim)
end


# TODO: Write the gradients and hessians for zero, constant and quadratic.
ϕ_zero(x) = 0.
∇ϕ_zero!(g, x) = begin
    g .= 0.
    return g
end

ϕ_constant(x) = 1.
∇ϕ_constant!(g, x) = begin
    g .= 0.
    return g
end


ϕ_quadratic(x::AbstractVector{T}) where T = dot(x, x)
function ∇ϕ_quadratic!(g::AbstractVector{T}, x::AbstractVector{T}) where T
    g .= 2. * x
    return g
end