const RANDOM_ACQUISITION = "Random"
const DOUBLE = 2
const JITTER = 1e-8
const NEWTON_SOLVE_TIME_LIMIT = .1
const MAX_DIMENSION = 30

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
        @eval $(fname)(x) = x[$j]
    end
    return nothing
end

macro generate_∇linear_phi(dim)
    # Evaluate the argument in the current module to get its value.
    dim_val = Base.eval(@__MODULE__, dim)
    for j in 1:dim_val
        fname = Symbol("∇linear_phi_", j, "!")
        @eval $(fname)(G, x) = (fill!(G, 0.0); G[$j] = 1.0; return G)
    end
    return nothing
end

macro generate_Hlinear_phi(dim)
    # Evaluate the argument in the current module to get its value.
    dim_val = Base.eval(@__MODULE__, dim)
    for j in 1:dim_val
        fname = Symbol("Hlinear_phi_", j, "!")
        @eval $(fname)(H, x) = (fill!(H, 0.0); return H)
    end
    return nothing
end

@generate_linear_phi MAX_DIMENSION
@generate_∇linear_phi MAX_DIMENSION
@generate_Hlinear_phi MAX_DIMENSION

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

function get_Hlinear_phi_functions(dim::Int)
    # Returns an NTuple of the first `dim` hessian functions.
    return ntuple(i -> getfield(@__MODULE__, Symbol("Hlinear_phi_", i, "!")), dim)
end

# TODO: Write the gradients and hessians for zero, constant and quadratic.
ϕ_zero(x) = 0.
∇ϕ_zero!(g, x) = begin
    g .= 0.
    return g
end
Hϕ_zero!(H, x) = begin
    H .= 0.
    return H    
end

ϕ_constant(x) = 1.
∇ϕ_constant!(g, x) = begin
    g .= 0.
    return g
end
Hϕ_constant!(H, x) = begin
    H .= 0.
    return H    
end

ϕ_quadratic(x) = dot(x, x)
∇ϕ_quadratic!(g, x) = begin
    g .= 2. * x
    return g
end
Hϕ_quadratic!(H, x) = begin
    H .= 2.
    return g
end
