@doc raw"""
    struct PolynomialBasisFunction{N, F<:NTuple{N, Function}, G<:NTuple{N, Function}, H<:NTuple{N, Function}} <: ParametricRepresentation

A representation of a polynomial basis function for use in parametric modeling, optimization, or approximation problems.
This struct encapsulates the basis functions and provides gradients and Hessians computed via automatic differentiation.

# Fields
- `basis_functions::F`: A tuple of functions representing the polynomial basis functions.
- `basis_gradients::G`: A tuple of functions representing the gradients of the basis functions.
- `basis_hessians::H`: A tuple of functions representing the Hessians of the basis functions.
"""
struct PolynomialBasisFunction{N, F<:NTuple{N, Function}, G<:NTuple{N, Function}, H<:NTuple{N, Function}} <: ParametricRepresentation
    basis_functions::F
    basis_gradients::G
    basis_hessians::H
end

function PolynomialBasisFunction(basis_functions::NTuple{N, Function}) where {N}
    basis_gradients = ntuple(i -> x -> ForwardDiff.gradient(basis_functions[i], x), N)
    basis_hessians  = ntuple(i -> x -> ForwardDiff.hessian(basis_functions[i], x), N)
    return PolynomialBasisFunction{N, typeof(basis_functions), typeof(basis_gradients), typeof(basis_hessians)}(basis_functions, basis_gradients, basis_hessians)
end

get_basis_functions(pbf::PolynomialBasisFunction) = pbf.basis_functions
get_basis_gradients(pbf::PolynomialBasisFunction) = pbf.basis_gradients
get_basis_hessians(pbf::PolynomialBasisFunction) = pbf.basis_hessians

Base.length(pbf::PolynomialBasisFunction) = length(pbf.basis_functions)

function eval_basis(pbf::PolynomialBasisFunction, x::AbstractVector{T}) where T <: Real
    # basis_functions = get_basis_functions(pbf)
    basis_evaluation = zeros(1, length(pbf.basis_functions))
    
    # for (i, bf) in enumerate(basis_functions)
    for i in 1:length(pbf)
        basis_evaluation[1, i] = pbf.basis_functions[i](x)
    end

    return basis_evaluation
end

# function eval_basis!(pbf::PolynomialBasisFunction, x::AbstractVector{T}, out::AbstractMatrix{T}) where T <: Real
function eval_basis!(pbf::PolynomialBasisFunction{N, F, G, H}, x::AbstractVector{T}, out::AbstractMatrix{T}) where {N, F, G, H, T <: Real}
    # basis_functions = get_basis_functions(pbf)
    
    @views begin
        for i in 1:length(pbf)
            out[1, i] = pbf.basis_functions[i](x)
        end
    end

    return nothing
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

function eval_basis!(pbf::PolynomialBasisFunction, X::AbstractMatrix{T}, out::AbstractMatrix{T}) where T <: Real
    d, N = size(X)
    basis_functions = get_basis_functions(pbf)

    @views begin
        for i in 1:length(basis_functions)
            bf = basis_functions[i]
            for j in 1:N
                out[j, i] = bf(X[:, j])
            end
        end
    end

    return nothing
end

function eval_∇basis(pbf::PolynomialBasisFunction, x::AbstractVector{T}) where T <: Real
    basis_functions = get_basis_gradients(pbf)
    basis_evaluation = zeros(length(x), length(basis_functions))

    for (i, ∇bf) in enumerate(basis_functions)
        basis_evaluation[:, i] = ∇bf(x)
    end

    return basis_evaluation
end

function eval_∇basis!(pbf::PolynomialBasisFunction, x::AbstractVector{T}, out::AbstractMatrix{T}) where T <: Real
    # basis_evaluation = zeros(length(x), length(basis_functions))

    for i in 1:length(pbf.basis_gradients)
        out[:, i] = pbf.basis_gradients[i](x)
    end

    return nothing
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

function eval_Hbasis!(pbf::PolynomialBasisFunction, x::AbstractVector{T}, out::AbstractArray{T}) where T <: Real
    for k in 1:length(pbf.basis_hessians)
        out[:, :, 1, k] = pbf.basis_hessians[k](x)
    end

    return nothing
end

(pbf::PolynomialBasisFunction)(x::AbstractVector{T}) where T <: Real = eval_basis(pbf, x)
(pbf::PolynomialBasisFunction)(X::AbstractMatrix{T}) where T <: Real = eval_basis(pbf, X)


struct PreallocatedContainers
    px
    ∇px
    Hpx
end

function PreallocatedContainers(pbf::PolynomialBasisFunction, dim::Int)
    return PreallocatedContainers(
        zeros(1, length(pbf.basis_functions)),
        zeros(dim, length(pbf.basis_functions)),
        zeros(dim, dim, 1, length(pbf.basis_functions))
    )
end