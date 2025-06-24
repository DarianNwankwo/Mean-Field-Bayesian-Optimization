# TODO: Include noise variance term in kernel hyperparameters
get_hyperparameters(sk::AbstractKernel) = sk.θ
Base.length(k::StationaryKernel) = length(get_hyperparameters(k))

@doc raw"""
A radial basis function is defined strictly in terms of the pairwise distance between
two locations. The representation below encodes this in terms of ρ = ||x - y||. The
attribute ψ is a function defined on this distance metric, that is, ψ(ρ). We also care
about the gradient and hessian with respect to this distance metric which we denote
as Dρ_ψ and Dρρ_ψ respectively.

One also cares about computing the gradient of ψ with respect to the radial basis
function's hyperparameters which we denote as ∇θ_ψ!. 
"""
struct RadialBasisFunction{N, T <: Real, K <: Function, D1 <: Function, D2 <: Function, D3 <: Function} <: StationaryKernel
    θ::MVector{N, T}
    ψ::K                   # concrete kernel function
    Dρ_ψ::D1              # derivative with respect to ρ
    Dρρ_ψ::D2             # second derivative with respect to ρ
    ∇θ_ψ!::D3              # derivative with respect to hyperparameters
end

function Base.show(io::IO, r::RadialBasisFunction{T}) where T
    print(io, "RadialBasisFunction{", T, "}")
end

@inline (rbf::RadialBasisFunction)(ρ::Real) = rbf.ψ(ρ, rbf.θ)
derivative(rbf::RadialBasisFunction) = rbf.Dρ_ψ
second_derivative(rbf::RadialBasisFunction) = rbf.Dρρ_ψ
hypersgradient(rbf::RadialBasisFunction) = rbf.∇θ_ψ!

function set_hyperparameters!(rbf::RadialBasisFunction, θ::Vector{T}) where T <: Real
    @views begin
        rbf.θ .= θ
    end

    return rbf
end


function Matern52_ψ(ρ::Real, θ::AbstractVector{<:Real})
    l = θ[1]
    v = θ[2]
    c = sqrt(5.0) / l
    s = c * ρ
    return v * (1 + s + s^2/3) * exp(-s)
end

function Matern52_Dρψ(ρ::Real, θ::AbstractVector{<:Real})
    l = θ[1]
    v = θ[2]
    c = sqrt(5.0) / l
    s = c * ρ
    # Derivative with respect to ρ: dψ/dρ = - v * (c * s * (1+s)/3)*exp(-s)
    return -v * (c * s * (1 + s)/3) * exp(-s)
end

function Matern52_Dρρψ(ρ::Real, θ::AbstractVector{<:Real})
    l = θ[1]
    v = θ[2]
    c = sqrt(5.0) / l
    s = c * ρ
    # Second derivative: d²ψ/dρ² = - v * (c^2)*exp(-s)*(1+s-s^2)/3
    return - v * (c^2) * exp(-s) * (1 + s - s^2)/3
end

# TODO: Update gradient of this wrt lengthscale and noise variance
function Matern52_∇θψ(G::AbstractVector{<:Real}, ρ::Real, θ::AbstractVector{<:Real})
    l = θ[1]
    v = θ[2]
    c = sqrt(5.0) / l
    s = c * ρ
    # Compute ∂ψ/∂l as (s + s^2)*exp(-s)*(sqrt(5)*ρ)/(3*l^2)
    G[1] = v * (s + s^2) * exp(-s) * (sqrt(5.0) * ρ) / (3 * l^2)
    G[2] = 2. * v * (1 + s + s^2/3) * exp(-s)
    return G
end

"""
    Matern52(θ=[1., 1.])

Construct a RadialBasisFunction representing the Matern 5/2 kernel using
the analytic functions defined in `Matern52_ψ`, `Matern52_Dρψ`,
`Matern52_Dρρψ`, and `Matern52_∇θψ`. The default hyperparameter vector is [1.].
"""
function Matern52(θ::AbstractVector{T} = [1., 1.]) where T <: Real
    n = length(θ)
    static_θ = MVector{n}(θ)
    return RadialBasisFunction{n,
                               T,
                               typeof(Matern52_ψ),
                               typeof(Matern52_Dρψ),
                               typeof(Matern52_Dρρψ),
                               typeof(Matern52_∇θψ)}(
        static_θ,
        Matern52_ψ,
        Matern52_Dρψ,
        Matern52_Dρρψ,
        Matern52_∇θψ
    )
end

"""
We evaluate our kernel given some normed distance ψ(ρ) where ρ = ||r||. We also
want to compute the gradient and hessian with respect to ρ. We care about the 
gradient with respect to the hyperparameters and we also want the mixed partials,
i.e. perturbations of the gradient with respect to the kernel hyperparameters:
∂/∂θ[ ψ'(ρ) ].
"""
eval_k(rbf::RadialBasisFunction, r::AbstractVector{T}) where T <: Real = rbf(norm(r))


function eval_∇k!(rbf::RadialBasisFunction, r::AbstractVector{T}, ∇k::AbstractVector) where T <: Real
    @views begin
        ρ = norm(r)
        if ρ == 0
            ∇k[:] = 0*r
            return ∇k
        end
        ∇ρ = r / ρ
        ∇k[:] = derivative(rbf)(ρ, rbf.θ) * ∇ρ
        return ∇k
    end
end


function eval_KXX!(
    rbf::RadialBasisFunction,
    X::AbstractMatrix{T1},
    # KXX::AbstractMatrix{T2},
    KXX::SubArray{T2, N2, A, I, L},
    diff::AbstractVector{T3}) where {T1 <: Real, T2 <: Real, T3 <: Real, N2, A, I, L}
    d, N = size(X)
    ψ0 = rbf(0.0)

    @inbounds @views begin
        for j = 1:N
            KXX[j,j] = ψ0
            for i = j+1:N
                @simd for k=1:d
                    diff[k] = X[k, i] - X[k, j]
                end
                norm_diff = sqrt(dot(diff, diff))
                KXX[i,j] = rbf(norm_diff)
                KXX[j,i] = KXX[i, j]
            end
        end
    end

    return KXX
end


function eval_KxX!(
    rbf::RadialBasisFunction,
    x::AbstractVector{<:Real},
    X::AbstractMatrix{<:Real},
    KxX::AbstractVector{<:Real},
    diff::AbstractVector{<:Real})
    d, N = size(X)
    
    @inbounds @views begin
        for i = 1:N
            @simd for k =1:d
                diff[k] = x[k] - X[k, i]
            end
            norm_diff = sqrt(dot(diff, diff))
            KxX[i] = rbf(norm_diff)
        end
    end

    return KxX
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
            norm_diff = sqrt(dot(diff, diff))
            ρ = norm_diff
            if ρ > 0
                diff ./= ρ
                diff .*= rbf.Dρ_ψ(ρ, rbf.θ)
                ∇KxX[:,j] .= diff
            end
        end
    end

    return ∇KxX
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
                δKij = eval_∇k!(rbf, diff, ∇k)' * (δdiff)
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
    diff::AbstractVector{T},
    δψij::AbstractVector{T}) where T <: Real
    @views begin
        d, N = size(X)
        
        for j = 1:N
            δKXX[j, j] = dot(hypersgradient(rbf)(δψij, 0.0, rbf.θ), δθ)
        end

        for j = 1:N
            for i = j+1:N
                @simd for k=1:d
                    diff[k] = X[k, i] - X[k, j]
                end
                norm_diff = sqrt(dot(diff, diff))

                δKij = dot(hypersgradient(rbf)(δψij, norm_diff, rbf.θ),  δθ)
                δKXX[i,j] = δKij
                δKXX[j,i] = δKXX[i,j]
            end
        end

        return δKXX
    end
end