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
struct RadialBasisFunction{T <: Real} <: StationaryKernel
    θ::Vector{T}
    ψ
    Dρ_ψ
    Dρρ_ψ
    ∇θ_ψ
end

function Base.show(io::IO, r::RadialBasisFunction{T}) where T
    print(io, "RadialBasisFunction{", T, "}")
end

(rbf::RadialBasisFunction)(ρ) = rbf.ψ(ρ)
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
function RadialBasisFunctionGeneric(k::Function, θ::Vector{T}) where T <: Real
    # Define the radial basis function ψ(ρ)
    ψ(ρ) = k(ρ, θ)
    Dρ_ψ, Dρρ_ψ, ∇θ_ψ = compute_derivatives(k, θ, ψ)
    
    # Return the constructed RadialBasisFunction with concrete types
    return RadialBasisFunction(θ, ψ, Dρ_ψ, Dρρ_ψ, ∇θ_ψ)
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
eval_k(rbf::RadialBasisFunction, r::Vector{T}) where T <: Real = rbf(norm(r))

function eval_∇k(rbf::RadialBasisFunction, r::Vector{T}) where T <: Real
    ρ = norm(r)
    if ρ == 0
        return 0*r
    end
    ∇ρ = r/ρ
    return derivative(rbf)(ρ)*∇ρ
end

function eval_Hk(rbf::RadialBasisFunction, r::Vector{T}) where T <: Real
    p = norm(r)
    if p > 0
        ∇p = r/p
        Dψr = derivative(rbf)(p)/p
        D2ψ = second_derivative(rbf)(p)
        return (D2ψ-Dψr)*∇p*∇p' + Dψr*I
    end
    return second_derivative(rbf)(p) * Matrix(I, length(r), length(r))
end

function eval_Dk(rbf::RadialBasisFunction, r::AbstractVector{T}) where T <: Real
    K = eval_k(rbf, r)
    ∇K = eval_∇k(rbf, r)
    HK = eval_Hk(rbf, r)
    
    return [K   -∇K'
            ∇K -HK]
end

function eval_KXX(rbf::RadialBasisFunction, X::AbstractMatrix{T}) where T <: Real
    d, N = size(X)
    KXX = zeros(N, N)
    ψ0 = rbf(0.0)

    @views begin
        for j = 1:N
            KXX[j,j] = ψ0
            for i = j+1:N
                Kij = rbf(norm(X[:,i]-X[:,j]))
                KXX[i,j] = Kij
                KXX[j,i] = Kij
            end
        end
    end

    return KXX
end

function eval_KXY(rbf::RadialBasisFunction, X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where T <: Real
    d1, N1 = size(X)
    d2, N2 = size(Y)
    KXY = zeros(N1, N2)

    @views begin
        for j = 1:N2
            for i = 1:N1
                Kij = rbf(norm(X[:, i] - Y[:, j]))
                KXY[i, j] = Kij
            end
        end
    end

    return KXY
end

function eval_KxX(rbf::RadialBasisFunction, x::AbstractVector{T}, X::AbstractMatrix{T}) where T <: Real
    d, N = size(X)
    KxX = zeros(N)
    
    @views begin
        for i = 1:N
            KxX[i] = rbf(norm(x-X[:,i]))
        end
    end

    return KxX
end

function eval_∇KxX(rbf::RadialBasisFunction, x::AbstractVector{T}, X::AbstractMatrix{T}) where T <: Real
    d, N = size(X)
    ∇KxX = zeros(d, N)
    
    @views begin
        for j = 1:N
            r = x-X[:,j]
            ρ = norm(r)
            if ρ > 0
                ∇KxX[:,j] = rbf.Dρ_ψ(ρ)*r/ρ
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

    @views begin
        for j = 1:N
            for i = j+1:N
                δKij = eval_∇k(rbf, X[:,i]-X[:,j])' * (δX[:,i]-δX[:,j])
                δKXX[i,j] = δKij
                δKXX[j,i] = δKij
            end
        end
    end

    return δKXX
end

function eval_δKxX(
    rbf::RadialBasisFunction,
    x::AbstractVector{T},
    X::AbstractMatrix{T},
    δX::AbstractMatrix{T}) where T <: Real
    d, N = size(X)
    δKxX = zeros(N)

    @views begin
        for j = 1:N
            δKxX[j] = eval_∇k(rbf, x-X[:,j])' * (-δX[:,j])
        end
    end

    return δKxX
end

function eval_δ∇KxX(
    rbf::RadialBasisFunction,
    x::AbstractVector{T},
    X::AbstractMatrix{T},
    δX::AbstractMatrix{T}) where T <: Real
    d, N = size(X)
    δ∇KxX = zeros(d, N)

    @views begin
        for j = 1:N
            δ∇KxX[:,j] = eval_Hk(rbf, x-X[:,j]) * (-δX[:,j])
        end
    end

    return δ∇KxX
end

function eval_Dθ_KXX(
    rbf::RadialBasisFunction,
    X::AbstractMatrix{T},
    δθ::AbstractVector{T}) where T <: Real
    d, N = size(X)
    δKXX = zeros(N, N)
    δψ0 = rbf.∇θ_ψ(0.0)' * δθ

    @views begin
        for j = 1:N
            δKXX[j,j] = δψ0
            for i = j+1:N
                δKij = rbf.∇θ_ψ(norm(X[:,i]-X[:,j]))' * δθ
                δKXX[i,j] = δKij
                δKXX[j,i] = δKij
            end
        end
    end

    return δKXX
end