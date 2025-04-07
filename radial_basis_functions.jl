get_hyperparameters(sk::AbstractKernel) = sk.θ

@doc raw"""
A radial basis function is defined strictly in terms of the pairwise distance between
two locations. The representation below encodes this in terms of ρ = ||x - y||. The
attribute ψ is a function defined on this distance metric, that is, ψ(ρ). We also care
about the gradient and hessian with respect to this distance metric which we denote
as Dρ_ψ and Dρρ_ψ respectively.

One also cares about computing the gradient of ψ with respect to the radial basis
function's hyperparameters which we denote as ∇θ_ψ. 
"""
struct RadialBasisFunction{T <: Real, K, D1, D2, D3} <: StationaryKernel
    θ::Vector{T}
    ψ::K                   # concrete kernel function
    Dρ_ψ::D1              # derivative with respect to ρ
    Dρρ_ψ::D2             # second derivative with respect to ρ
    ∇θ_ψ::D3              # derivative with respect to hyperparameters
end


function Base.show(io::IO, r::RadialBasisFunction{T}) where T
    print(io, "RadialBasisFunction{", T, "}")
end

(rbf::RadialBasisFunction)(ρ::Real) = rbf.ψ(ρ)
derivative(rbf::RadialBasisFunction) = rbf.Dρ_ψ
second_derivative(rbf::RadialBasisFunction) = rbf.Dρρ_ψ
hypersgradient(rbf::RadialBasisFunction) = rbf.∇θ_ψ

function set_hyperparameters!(rbf::RadialBasisFunction, θ::Vector{T}) where T <: Real
    @views begin
        rbf.θ .= θ
    end

    return rbf
end

"""
A generic way of constructing a radial basis functions with a given kernel
using automatic differentiation. The kernel should be a function of the normed
distance ρ and the hyperparameter vector θ. The kernel should be
differentiable with respect to ρ and θ. The kernel should be positive
definite.
"""
function compute_derivatives(k::Function, θ::Vector{T}, ψ::Function) where T
    Dρ_ψ(ρ) = ForwardDiff.derivative(ψ, ρ)        # Derivative wrt ρ
    Dρρ_ψ(ρ) = ForwardDiff.derivative(Dρ_ψ, ρ)    # Second derivative wrt ρ
    ∇θ_ψ(ρ) = ForwardDiff.gradient(θ -> k(ρ, θ), θ)  # Gradient wrt θ
    return Dρ_ψ, Dρρ_ψ, ∇θ_ψ  # Return all derivatives
end

"""
A stable generic constructor for the struct RadialBasisFunction defined above. It
computes all the necessary attributes provided the user gives a kernel function
k, in terms of the normed distance, and hyperparameter vector θ.
"""
function RadialBasisFunctionGeneric(k::K, θ::Vector{T}) where {K, T <: Real}
    # Compute derivatives using your existing method
    ψ(ρ) = k(ρ, θ)
    Dρ_ψ, Dρρ_ψ, ∇θ_ψ = compute_derivatives(k, θ, ψ)
    
    # Construct the RadialBasisFunction with the concrete kernel function k.
    return RadialBasisFunction{T, typeof(ψ), typeof(Dρ_ψ), typeof(Dρρ_ψ), typeof(∇θ_ψ)}(θ, ψ, Dρ_ψ, Dρρ_ψ, ∇θ_ψ)
end

"""
Some common kernels of interest defined as required and passed to our generic
constructor.
"""
function Matern52(θ=[1.])
    function k(ρ, θ)
        l = θ[1]
        c = sqrt(5.0) / l
        s = c*ρ
        return (1+s*(1+s/3.0))*exp(-s)
    end
    return RadialBasisFunctionGeneric(k, θ)
end

function Matern32(θ=[1.])
    function k(ρ, θ)
        l = θ[1]
        c = sqrt(3.0) / l
        s = c*ρ
        return (1+s)*exp(-s)
    end
    return RadialBasisFunctionGeneric(k, θ)
end

function Matern12(θ=[1.])
    function k(ρ, θ)
        l = θ[1]
        c = 1.0 / l
        s = c*ρ
        return exp(-s)
    end
    return RadialBasisFunctionGeneric(k, θ)
end

function SquaredExponential(θ=[1.])
    function k(ρ, θ)
        l = θ[1]
        return exp(-ρ^2/(2*l^2))
    end
    return RadialBasisFunctionGeneric(k, θ)
end

function Periodic(θ=[1., 1.])
    function k(ρ, θ)
        return exp(-2 * sin(pi * ρ / θ[2]) ^ 2 / θ[1] ^ 2)
    end
    return RadialBasisFunctionGeneric(k, θ)
end

"""
We evaluate our kernel given some normed distance ψ(ρ) where ρ = ||r||. We also
want to compute the gradient and hessian with respect to ρ. We care about the 
gradient with respect to the hyperparameters and we also want the mixed partials,
i.e. perturbations of the gradient with respect to the kernel hyperparameters:
∂/∂θ[ ψ'(ρ) ].
"""
eval_k(rbf::RadialBasisFunction, r::AbstractVector{T}) where T <: Real = rbf(norm(r))

function eval_∇k(rbf::RadialBasisFunction, r::AbstractVector{T}) where T <: Real
    ρ = norm(r)
    if ρ == 0
        return 0*r
    end
    ∇ρ = r/ρ
    return derivative(rbf)(ρ)*∇ρ
end

function eval_∇k!(rbf::RadialBasisFunction, r::AbstractVector{T}, ∇k::AbstractVector) where T <: Real
    @views begin
        ρ = norm(r)
        if ρ == 0
            ∇k[:] = 0*r
            return ∇k
        end
        ∇ρ = r / ρ
        ∇k[:] = derivative(rbf)(ρ) * ∇ρ
        return ∇k
    end
end

function eval_Hk(rbf::RadialBasisFunction, r::Vector{T}) where T <: Real
    ρ = norm(r)
    if ρ > 0
        ∇ρ = r/ρ
        Dψr = derivative(rbf)(ρ)/ρ
        D2ψ = second_derivative(rbf)(ρ)
        return (D2ψ-Dψr)*∇ρ*∇ρ' + Dψr*I
    end
    return second_derivative(rbf)(ρ) * Matrix(I, length(r), length(r))
end

function eval_Hk!(rbf::RadialBasisFunction, r::AbstractVector{T}, Hk::AbstractMatrix{T}) where T <: Real
    @views begin
        ρ = norm(r)
        if ρ > 0
            ∇ρ = r/ρ
            Dψr = derivative(rbf)(ρ)/ρ
            D2ψ = second_derivative(rbf)(ρ)
            Hk[:, :] = (D2ψ-Dψr)*∇ρ*∇ρ' + Dψr*I
            return Hk
        end
        Hk[:, :] = second_derivative(rbf)(ρ) * Matrix(I, length(r), length(r))
        return Hk
    end
end

function eval_KXX(rbf::RadialBasisFunction, X::AbstractMatrix{T}) where T <: Real
    d, N = size(X)
    KXX = zeros(N, N)
    ψ0 = rbf(0.0)
    diff = zeros(T, d)

    @inbounds @views begin
        for j = 1:N
            KXX[j,j] = ψ0
            for i = j+1:N
                @simd for k=1:d
                    diff[k] = X[k, i] - X[k, j]
                end
                KXX[i,j] = rbf(norm(diff))
                KXX[j,i] = KXX[i, j]
            end
        end
    end

    return KXX
end
(rbf::RadialBasisFunction)(X::AbstractMatrix{T}) where T <: Real = eval_KXX(rbf, X)

function eval_KXX!(rbf::RadialBasisFunction, X::AbstractMatrix{T}, KXX::AbstractMatrix{T}, diff::AbstractVector{T}) where T <: Real
    d, N = size(X)
    ψ0 = rbf(0.0)

    @inbounds @views begin
        for j = 1:N
            KXX[j,j] = ψ0
            for i = j+1:N
                @simd for k=1:d
                    diff[k] = X[k, i] - X[k, j]
                end
                KXX[i,j] = rbf(norm(diff))
                KXX[j,i] = KXX[i, j]
            end
        end
    end

    return KXX
end
(rbf::RadialBasisFunction)(
    X::AbstractMatrix{T},
    KXX::AbstractMatrix{T},
    diff::AbstractVector{T}
) where T <: Real = eval_KXX!(rbf, X, KXX, diff)

function eval_KXY(rbf::RadialBasisFunction, X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where T <: Real
    d, N1 = size(X)
    _, N2 = size(Y)
    KXY = zeros(N1, N2)
    diff = zeros(T, d1)

    @inbounds @views begin
        for j = 1:N2
            for i = 1:N1
                @simd for k=1:d
                    diff[k] = X[k, i] - Y[k, j]
                end
                KXY[i, j] = rbf(norm(diff))
            end
        end
    end

    return KXY
end
(rbf::RadialBasisFunction)(X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where T <: Real = eval_KXY(rbf, X, Y)

function eval_KXY!(
    rbf::RadialBasisFunction,
    X::AbstractMatrix{T},
    Y::AbstractMatrix{T},
    KXY::AbstractMatrix{T},
    diff::AbstractVector{T}) where T <: Real
    d, N1 = size(X)
    _, N2 = size(Y)

    @inbounds @views begin
        for j = 1:N2
            for i = 1:N1
                @simd for k=1:d
                    diff[k] = X[k, i] - Y[k, j]
                end
                KXY[i, j] = rbf(norm(diff))
            end
        end
    end

    return KXY
end

function eval_KxX(rbf::RadialBasisFunction, x::AbstractVector{T}, X::AbstractMatrix{T}) where T <: Real
    d, N = size(X)
    KxX = zeros(T, N)
    diff = zeros(T, d)
    
    @inbounds @views begin
        for i = 1:N
            @simd for k =1:d
                diff[k] = x[k] - X[k, i]
            end
            KxX[i] = rbf(norm(diff))
        end
    end

    return KxX
end
(rbf::RadialBasisFunction)(x::AbstractVector{T}, X::AbstractMatrix{T}) where T <: Real = eval_KxX(rbf, x, X)

function eval_KxX!(
    rbf::RadialBasisFunction,
    x::AbstractVector{T},
    X::AbstractMatrix{T},
    KxX::AbstractVector{T},
    diff::AbstractVector{T}) where T <: Real
    d, N = size(X)
    
    @inbounds @views begin
        for i = 1:N
            @simd for k =1:d
                diff[k] = x[k] - X[k, i]
            end
            KxX[i] = rbf(norm(diff))
        end
    end

    return KxX
end

function eval_∇KxX(rbf::RadialBasisFunction, x::AbstractVector{T}, X::AbstractMatrix{T}) where T <: Real
    d, N = size(X)
    ∇KxX = zeros(d, N)
    diff = zeros(T, d)
    
    @inbounds @views begin
        for j = 1:N
            @simd for k=1:d
                diff[k] = x[k] - X[k, j]
            end
            # r = x-X[:,j]
            ρ = norm(diff)
            if ρ > 0
                ∇KxX[:,j] = rbf.Dρ_ψ(ρ)*diff/ρ
            end
        end
    end

    return ∇KxX
end

function eval_∇KxX!(
    rbf::RadialBasisFunction,
    x::AbstractVector{T},
    X::AbstractMatrix{T},
    ∇KxX::AbstractMatrix{T},
    diff::AbstractVector{T}) where T <: Real
    d, N = size(X)
    
    @inbounds @views begin
        for j = 1:N
            @simd for k=1:d
                diff[k] = x[k] - X[k, j]
            end
            ρ = norm(diff)
            if ρ > 0
                ∇KxX[:,j] = rbf.Dρ_ψ(ρ)*diff/ρ
            end
        end
    end

    return ∇KxX
end

function eval_δKXX(
    rbf::RadialBasisFunction,
    X::AbstractMatrix{T},
    δX::AbstractMatrix{T}) where T <: Real
    d, N = size(X)
    δKXX = zeros(N, N)
    diff = zeros(T, d)
    δdiff = zeros(T, d)

    @inbounds @views begin
        for j = 1:N
            for i = j+1:N
                @simd for k=1:d
                    diff[k] = X[k, i] - X[k, j]
                    δdiff[k] = δX[k, i] - δX[k, j]
                end
                δKij = eval_∇k(rbf, diff)' * (δdiff)
                δKXX[i,j] = δKij
                δKXX[j,i] = δKij
            end
        end
    end

    return δKXX
end


function eval_δKXX!(
    rbf::RadialBasisFunction,
    X::AbstractMatrix{T},
    δX::AbstractMatrix{T},
    δKXX::AbstractMatrix{T},
    diff::AbstractVector{T},
    δdiff::AbstractVector{T},
    ∇k::AbstractVector{T}) where T <: Real
    d, N = size(X)

    @inbounds @views begin
        for j = 1:N
            for i = j+1:N
                @simd for k=1:d
                    diff[k] = X[k, i] - X[k, j]
                    δdiff[k] = δX[k, i] - δX[k, j]
                end
                # δKij = eval_∇k!(rbf, diff, ∇k)' * (δdiff)
                δKij = 1.
                δKXX[i,j] = δKij
                δKXX[j,i] = δKij
            end
        end
    end

    return δKXX
end


# function eval_δKxX(
#     rbf::RadialBasisFunction,
#     x::AbstractVector{T},
#     X::AbstractMatrix{T},
#     δX::AbstractMatrix{T}) where T <: Real
#     d, N = size(X)
#     δKxX = zeros(N)
#     diff = zeros(T, d)

#     @inbounds @views begin
#         for j = 1:N
#             @simd for k=1:d
#                 diff[k] = x[k] - X[k, j]
#             end
#             δKxX[j] = eval_∇k(rbf, diff)' * (-δX[:,j])
#         end
#     end

#     return δKxX
# end

# function eval_δ∇KxX(
#     rbf::RadialBasisFunction,
#     x::AbstractVector{T},
#     X::AbstractMatrix{T},
#     δX::AbstractMatrix{T}) where T <: Real
#     d, N = size(X)
#     δ∇KxX = zeros(d, N)
#     diff = zeros(T, d)

#     @views begin
#         for j = 1:N
#             @simd for k=1:d
#                 diff[k] = x[k] - X[k, j]
#             end
#             δ∇KxX[:,j] = eval_Hk(rbf, diff) * (-δX[:,j])
#         end
#     end

#     return δ∇KxX
# end

function eval_Dθ_KXX(
    rbf::RadialBasisFunction,
    X::AbstractMatrix{T},
    δθ::AbstractVector{T}) where T <: Real
    d, N = size(X)
    δKXX = zeros(N, N)
    δψ0 = rbf.∇θ_ψ(0.0)' * δθ
    diff = zeros(T, d)

    @views begin
        for j = 1:N
            δKXX[j,j] = δψ0
            for i = j+1:N
                @simd for k=1:d
                    diff[k] = X[k, i] - X[k, j]
                end
                δKij = rbf.∇θ_ψ(norm(diff))' * δθ
                δKXX[i,j] = δKij
                δKXX[j,i] = δKij
            end
        end
    end

    return δKXX
end


function eval_Dθ_KXX!(
    rbf::RadialBasisFunction,
    X::AbstractMatrix{T},
    δθ::AbstractVector{T},
    δKXX::AbstractMatrix{T},
    diff::AbstractVector{T}) where T <: Real
    d, N = size(X)
    δψ0 = rbf.∇θ_ψ(0.0)' * δθ

    @views begin
        for j = 1:N
            δKXX[j,j] = δψ0
            for i = j+1:N
                @simd for k=1:d
                    diff[k] = X[k, i] - X[k, j]
                end
                δKij = rbf.∇θ_ψ(norm(diff))' * δθ
                # δKij = 1.
                δKXX[i,j] = δKij
                δKXX[j,i] = δKij
            end
        end

        return δKXX
    end
end