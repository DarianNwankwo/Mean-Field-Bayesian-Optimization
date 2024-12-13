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

function eval_basis(pbf::PolynomialBasisFunction, x::Vector{T}) where T <: Real
    basis_functions = get_basis_functions(pbf)
    basis_evaluation = zeros(length(basis_functions))
    
    for (i, bf) in enumerate(basis_functions)
        basis_evaluation[i] = bf(x)
    end

    return basis_evaluation
end

function eval_basis(pbf::PolynomialBasisFunction, X::Matrix{T}) where T <: Real
    d, N = size(X)
    basis_functions = get_basis_functions(pbf)
    basis_evaluations = zeros(N, length(basis_functions))

    for i in 1:N
        basis_evaluations[i, :] = eval_basis(pbf, X[:, i])
    end

    return basis_evaluations
end

(pbf::PolynomialBasisFunction)(x::Vector{T}) where T <: Real = eval_basis(pbf, x)
(pbf::PolynomialBasisFunction)(X::Matrix{T}) where T <: Real = eval_basis(pbf, X)

function eval_polynomial(pbf::PolynomialBasisFunction, x::Vector{T1}, c::Vector{T2}) where {T1 <: Real, T2 <: Real}
    basis_evaluation = eval_basis(pbf, x)
    return dot(basis_evaluation, c)
end

function eval_polynomial(pbf::PolynomialBasisFunction, X::Matrix{T1}, c::Vector{T2}) where {T1 <: Real, T2 <: Real}
    basis_evaluations = eval_basis(pbf, X)
    return X*c
end

(pbf::PolynomialBasisFunction)(x::Vector{T1}, c::Vector{T2}) where {T1 <: Real, T2 <: Real} = eval_polynomial(pbf, x, c)
(pbf::PolynomialBasisFunction)(X::Matrix{T1}, c::Vector{T2}) where {T1 <: Real, T2 <: Real} = eval_polynomial(pbf, X, c)