@doc raw"""
    struct PolynomialBasisFunction <: ParametricRepresentation

A representation of a polynomial basis function for use in parametric modeling, optimization, or approximation problems.
This struct encapsulates the basis functions and provides gradients and Hessians computed via automatic differentiation.

# Fields
- `basis_functions::Vector{<:Function}`: A vector of functions representing the polynomial basis functions.
    Each function takes input values (e.g., scalars or vectors) and returns the corresponding basis function value.
- `basis_gradients::Vector{<:Function}`: A vector of functions representing the gradients of the basis functions.
    These gradients are automatically computed using automatic differentiation tools.
- `basis_hessians::Vector{<:Function}`: A vector of functions representing the Hessians of the basis functions.
    These Hessians are automatically computed using automatic differentiation tools.
"""
struct PolynomialBasisFunction <: ParametricRepresentation
    basis_functions::Vector{<:Function}
    basis_gradients::Vector{<:Function}
    basis_hessians::Vector{<:Function}
end

# Constructor that takes a list of basis functions and computes gradients/hessians
function PolynomialBasisFunction(basis_functions::Vector{<:Function})
    basis_gradients = [x -> ForwardDiff.gradient(bf, x) for bf in basis_functions]
    basis_hessians  = [x -> ForwardDiff.hessian(bf, x) for bf in basis_functions]
    return PolynomialBasisFunction(basis_functions, basis_gradients, basis_hessians)
end

get_basis_functions(pbf::PolynomialBasisFunction) = pbf.basis_functions
get_basis_gradients(pbf::PolynomialBasisFunction) = pbf.basis_gradients
get_basis_hessians(pbf::PolynomialBasisFunction) = pbf.basis_hessians

Base.length(pbf::PolynomialBasisFunction) = length(get_basis_functions(pbf))

function eval_basis(pbf::PolynomialBasisFunction, x::AbstractVector{T}) where T <: Real
    basis_functions = get_basis_functions(pbf)
    basis_evaluation = zeros(1, length(basis_functions))
    
    for (i, bf) in enumerate(basis_functions)
        basis_evaluation[1, i] = bf(x)
    end

    return basis_evaluation
end

function eval_basis(pbf::PolynomialBasisFunction, X::AbstractMatrix{T}) where T <: Real
    d, N = size(X)
    basis_functions = get_basis_functions(pbf)
    basis_evaluations = zeros(N, length(basis_functions))

    for i in 1:N
        basis_evaluations[i, :] = eval_basis(pbf, X[:, i])
    end

    return basis_evaluations
end

function eval_∇basis(pbf::PolynomialBasisFunction, x::AbstractVector{T}) where T <: Real
    basis_functions = get_basis_gradients(pbf)
    basis_evaluation = zeros(length(x), length(basis_functions))

    for (i, ∇bf) in enumerate(basis_functions)
        basis_evaluation[:, i] = ∇bf(x)
    end

    return basis_evaluation
end

function eval_∇basis(pbf::PolynomialBasisFunction, X::AbstractMatrix{T}) where T <: Real
    d, N = size(X)
    basis_functions = get_basis_gradients(pbf)
    basis_evaluations = [zeros(d, length(basis_functions)) for i in 1:N]

    for i in 1:N
        basis_evaluations[i][:, :] = eval_∇basis(pbf, X[:, i])
    end

    return basis_evaluations
end

function eval_Hbasis(pbf::PolynomialBasisFunction, x::AbstractVector{T}) where T <: Real
    basis_functions = get_basis_hessians(pbf)
    dim = length(x)
    # basis_evaluation = [zeros(dim, dim) for i in 1:length(basis_functions)]
    basis_evaluation = zeros(dim, dim, 1, length(basis_functions))

    for (k, Hbf) in enumerate(basis_functions)
        basis_evaluation[:, :, 1, k] = Hbf(x)
    end

    return basis_evaluation
end

(pbf::PolynomialBasisFunction)(x::AbstractVector{T}) where T <: Real = eval_basis(pbf, x)
(pbf::PolynomialBasisFunction)(X::AbstractMatrix{T}) where T <: Real = eval_basis(pbf, X)