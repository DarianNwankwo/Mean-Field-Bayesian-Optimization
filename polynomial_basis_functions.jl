@doc raw"""
    struct PolynomialBasisFunction{N, F<:NTuple{N, Function}, G<:NTuple{N, Function}, H<:NTuple{N, Function}} <: ParametricRepresentation

A representation of a polynomial basis function for use in parametric modeling, optimization, or approximation problems.
This struct encapsulates the basis functions and provides gradients and Hessians computed via automatic differentiation.

# Fields
- `basis_functions::F`: A tuple of functions representing the polynomial basis functions.
- `basis_gradients::G`: A tuple of functions representing the gradients of the basis functions.
"""
# struct PolynomialBasisFunction <: ParametricRepresentation
#     basis_functions::NTuple{N, Function} where N
#     basis_gradients::NTuple{N, Function} where N
# end
struct PolynomialBasisFunction{N, BF<:NTuple{N,Function}, BG<:NTuple{N,Function}} <: ParametricRepresentation
    basis_functions :: BF
    basis_gradients  :: BG
end

# function PolynomialBasisFunction(basis_functions::NTuple{N, Function}) where {N}
#     basis_gradients = ntuple(i -> (g, x) -> begin
#         g .= ForwardDiff.gradient(basis_functions[i], x)
#         return g
#     end, N)
#     return PolynomialBasisFunction(
#         basis_functions,
#         basis_gradients
#     )
# end
function PolynomialBasisFunction(basis_functions::NTuple{N, Function}) where {N}
    basis_gradients = ntuple(i -> (g, x) -> begin
        ForwardDiff.gradient!(g, basis_functions[i], x)
        return g
    end, N)
    return PolynomialBasisFunction(
        basis_functions,
        basis_gradients
    )
end

get_basis_functions(pbf::PolynomialBasisFunction) = pbf.basis_functions
get_basis_gradients(pbf::PolynomialBasisFunction) = pbf.basis_gradients


Base.length(pbf::PolynomialBasisFunction) = length(pbf.basis_functions)


# function eval_basis!(pbf::PolynomialBasisFunction, x::AbstractVector{T}, out::AbstractMatrix{T}) where T <: Real
function eval_basis!(pbf::PolynomialBasisFunction, x::AbstractVector{T}, out::AbstractMatrix{T}) where T <: Real    
    @views begin
        for i in 1:length(pbf)
            bf = pbf.basis_functions[i]
            out[1, i] = bf(x)
        end
    end

    return out
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

    return out
end

function eval_∇basis!(pbf::PolynomialBasisFunction, x::AbstractVector{T}, out::AbstractMatrix) where T
    # basis_evaluation = zeros(length(x), length(basis_functions))

    # TODO: Replace with non-allocating inplace operations for assigning
    # values to matrix
    for i in 1:length(pbf.basis_gradients)
        # out[:, i] = pbf.basis_gradients[i](x)
        pbf.basis_gradients[i]((@view out[:, i]), x)
    end

    return out
end