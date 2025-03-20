# Useful functions for grabbing desired attributes on our structs
get_kernel(afs::AbstractSurrogate) = afs.ψ
get_covariates(afs::AbstractSurrogate) = afs.X
get_observations(afs::AbstractSurrogate) = afs.y
get_decision_rule(afs::AbstractSurrogate) = afs.g
get_coefficients(afs::AbstractSurrogate) = afs.d
get_cholesky(as::AbstractSurrogate) = as.L
get_covariance(as::AbstractSurrogate) = as.K
get_capacity(s::AbstractSurrogate) = s.capacity
increment!(s::AbstractSurrogate) = s.observed += 1
get_observed(s::AbstractSurrogate) = s.observed
is_full(s::AbstractSurrogate) = get_observed(s) == get_capacity(s)
get_active_covariates(s::AbstractSurrogate) = @view get_covariates(s)[:, 1:get_observed(s)]
get_active_cholesky(s::AbstractSurrogate) = @view get_cholesky(s)[1:get_observed(s), 1:get_observed(s)]
get_active_covariance(s::AbstractSurrogate) = @view get_covariance(s)[1:get_observed(s), 1:get_observed(s)]
get_active_observations(s::AbstractSurrogate) = @view get_observations(s)[1:get_observed(s)]
get_active_coefficients(s::AbstractSurrogate) = @view get_coefficients(s)[1:get_observed(s)]
get_observation_noise(s::AbstractSurrogate) = s.observation_noise
set_decision_rule!(s::AbstractSurrogate, g::DecisionRule) = s.g = g


"""
The model we consider consists of some parametric and nonparametric component, i.e.
f(x) ∼ P(x)c + Z(x) where Z ∼ GP(0, k_h), covariance function k: Ω × Ω → R, multivariate 
normal c ∼ N(0, Σ) with Σ = (1/ϵ)Σ_{ref}^2, and P: Ω → R^m.

We need to maintain information about the covariance function, which is our kernel,
and the mechanism that allows us to construct our linear system given some polynomial
basis.

We also maintain how many the number of observations and our model's capacity, for computational
considerations. We preallocate, for memory purposes, ahead of time. In the event we reach capacity,
we double our model's capacity.
"""
mutable struct HybridSurrogate{RBF<:StationaryKernel,PBF<:ParametricRepresentation} <: AbstractSurrogate
    ψ::RBF
    ϕ::PBF
    X::Matrix{Float64} # Covariates
    P::Matrix{Float64} # Parametric term design matrix
    K::Matrix{Float64} # Covariance matrix for Gaussian process
    L::LowerTriangular{Float64,Matrix{Float64}} # Cholesky factorization of covariance matrix
    y::Vector{Float64}
    d::Vector{Float64} # Coefficients for Gaussian process
    λ::Vector{Float64} # Coefficients for parametric/trend term
    observation_noise::Float64
    g::DecisionRule
    observed::Int
    capacity::Int
end

get_parametric_basis_matrix(as::HybridSurrogate) = as.P
get_active_parametric_basis_matrix(s::HybridSurrogate) = @view get_parametric_basis_matrix(s)[1:get_observed(s), :]
get_parametric_basis_function(s::HybridSurrogate) = s.ϕ
get_parametric_component_coefficients(afs::HybridSurrogate) = afs.λ

mutable struct Surrogate{RBF<:StationaryKernel} <: AbstractSurrogate
    ψ::RBF
    X::Matrix{Float64}
    K::Matrix{Float64}
    L::LowerTriangular{Float64,Matrix{Float64}}
    y::Vector{Float64}
    d::Vector{Float64}
    observation_noise::Float64
    g::DecisionRule
    observed::Int
    capacity::Int
end


# Define the custom show method for Surrogate
function Base.show(io::IO, s::HybridSurrogate{RBF}) where {RBF}
    print(io, "HybridSurrogate{RBF = ")
    show(io, s.ψ)    # Use the show method for RBF
    print(io, "}")
end

get_decision_rule(s::AbstractSurrogate) = s.g
set_decision_rule!(s::AbstractSurrogate, g::DecisionRule) = s.g = g

@doc raw"""
    coefficient_solve(
        KXX::AbstractMatrix{T1},
        PX::AbstractMatrix{T2},
        y::AbstractVector{T3}
    ) where {T1 <: Real, T2 <: Real, T3 <: Real}

Solves a system of linear equations involving a block matrix structure to
compute coefficients `d` and `λ`. The block matrix equation has the following
form:

    [KXX   PX]   [d]   =   [y]
    [PX^T   0]   [λ]       [0]

In this equation:
- `KXX` is a square matrix of size k_dim by k_dim.
- `PX` is a rectangular matrix of size k_dim by p_dim.
- `y` is a vector of length k_dim.
- `d` is a vector of length k_dim, representing the solution coefficients for
    the top block of the system.
- `λ` is a vector of length p_dim, representing the solution coefficients for
    the bottom block of the system.

# Arguments
- `KXX::AbstractMatrix{T1}`: A square matrix of size k_dim by k_dim,
    representing part of the block structure.
- `PX::AbstractMatrix{T2}`: A rectangular matrix of size k_dim by p_dim,
    representing the coupling between the blocks.
- `y::AbstractVector{T3}`: A vector of length k_dim, representing the right-hand
    side of the system for the top block.

# Returns
- `d`: A vector of length k_dim, representing the solution coefficients for the
    top block of the system.
- `λ`: A vector of length p_dim, representing the solution coefficients for the
    bottom block of the system.
"""
function coefficient_solve(
    KXX::AbstractMatrix{T1},
    PX::AbstractMatrix{T2},
    y::AbstractVector{T3}) where {T1<:Real,T2<:Real,T3<:Real}
    p_dim, k_dim = size(PX, 2), size(KXX, 2)
    A = [KXX PX;
        PX' zeros(p_dim, p_dim)]
    F = lu(A)
    b = [y; zeros(p_dim)]
    w = F \ b
    d, λ = w[1:k_dim], w[k_dim+1:end]
    return d, λ
end

"""
Preallocate a covariate matrix of size (d x capacity) and assigned the first N
columns to be the given covariates.
"""
function HybridSurrogate(
    ψ::RadialBasisFunction,
    ϕ::PolynomialBasisFunction,
    X::Matrix{T},
    y::Vector{T};
    capacity::Int=DEFAULT_CAPACITY,
    decision_rule::AbstractDecisionRule=EI(),
    observation_noise::T=1e-6) where {T<:Real}
    @assert length(y) <= capacity "Capacity must be >= number of observations."
    d, N = size(X)

    """
    Preallocate a matrix for covariates of size d x capacity where capacity is the maximum
    number of observations.
    """
    preallocated_X = zeros(d, capacity)
    preallocated_X[:, 1:N] = X

    """
    Preallocate a matrix for the matrix that represents the polynomial basis evaluation of 
    our observations of size capacity x m where m is the dimensionality of our basis vector.
    """
    preallocated_P = zeros(capacity, length(ϕ))
    PX = eval_basis(ϕ, X)
    preallocated_P[1:N, 1:length(ϕ)] = PX

    """
    Preallocate a covariance matrix of size d x capacity
    """
    preallocated_K = zeros(capacity, capacity)
    KXX = eval_KXX(ψ, X) + (JITTER + observation_noise) * I
    preallocated_K[1:N, 1:N] = KXX

    """
    Preallocate a matrix for the cholesky factorization of size d x capacity
    """
    preallocated_L = LowerTriangular(zeros(capacity, capacity))
    preallocated_L[1:N, 1:N] = cholesky(
        Hermitian(
            preallocated_K[1:N, 1:N]
        )
    ).L

    """
    Linear system solve for learning coefficients of stochastic component and parametric
    component. 
    """
    d, λ = coefficient_solve(KXX, PX, y)

    preallocated_d = zeros(capacity)
    preallocated_d[1:length(d)] = d

    λ_polynomial = zeros(length(λ))
    λ_polynomial[:] = λ

    preallocated_y = zeros(capacity)
    preallocated_y[1:N] = y

    observed = length(y)

    return HybridSurrogate(
        ψ,
        ϕ,
        preallocated_X,
        preallocated_P,
        preallocated_K,
        preallocated_L,
        preallocated_y,
        preallocated_d,
        λ_polynomial,
        observation_noise,
        decision_rule,
        observed,
        capacity
    )
end

# Constructor that expects observations and covariates later
function HybridSurrogate(
    ψ::RadialBasisFunction,
    ϕ::PolynomialBasisFunction;
    dim::Int,
    capacity::Int=DEFAULT_CAPACITY,
    decision_rule::AbstractDecisionRule=EI(),
    observation_noise::T=1e-6) where {T<:Real}
    preallocated_X = zeros(dim, capacity)
    preallocated_P = zeros(capacity, length(ϕ))
    preallocated_K = zeros(capacity, capacity)
    preallocated_L = LowerTriangular(zeros(capacity, capacity))
    preallocated_d = zeros(capacity)
    λ_polynomial = zeros(length(ϕ))
    preallocated_y = zeros(capacity)
    observed = 0

    return HybridSurrogate(
        ψ,
        ϕ,
        preallocated_X,
        preallocated_P,
        preallocated_K,
        preallocated_L,
        preallocated_y,
        preallocated_d,
        λ_polynomial,
        observation_noise,
        decision_rule,
        observed,
        capacity
    )
end

function Surrogate(
    ψ::RadialBasisFunction,
    X::Matrix{T},
    y::Vector{T};
    capacity::Int=DEFAULT_CAPACITY,
    decision_rule::AbstractDecisionRule=EI(),
    observation_noise::T=1e-6) where {T<:Real}
    @assert length(y) <= capacity "Capacity must be >= number of observations."
    d, N = size(X)

    preallocated_X = zeros(d, capacity)
    preallocated_X[:, 1:N] = X

    preallocated_K = zeros(capacity, capacity)
    preallocated_K[1:N, 1:N] = eval_KXX(ψ, X) + (JITTER + observation_noise) * I

    preallocated_L = LowerTriangular(zeros(capacity, capacity))
    preallocated_L[1:N, 1:N] = cholesky(
        Hermitian(
            preallocated_K[1:N, 1:N]
        )
    ).L

    preallocated_d = zeros(capacity)
    preallocated_d[1:N] = preallocated_L[1:N, 1:N]' \ (preallocated_L[1:N, 1:N] \ y)

    preallocated_y = zeros(capacity)
    preallocated_y[1:N] = y

    return Surrogate(
        ψ,
        preallocated_X,
        preallocated_K,
        preallocated_L,
        preallocated_y,
        preallocated_d,
        observation_noise,
        decision_rule,
        length(y),
        capacity
    )
end

function Surrogate(
    ψ::RadialBasisFunction;
    dim::Int,
    capacity::Int=DEFAULT_CAPACITY,
    decision_rule::AbstractDecisionRule=EI(),
    observation_noise::T=1e-6) where {T<:Real}
    preallocated_X = zeros(dim, capacity)
    preallocated_K = zeros(capacity, capacity)
    preallocated_L = LowerTriangular(zeros(capacity, capacity))
    preallocated_d = zeros(capacity)
    preallocated_y = zeros(capacity)

    return Surrogate(
        ψ,
        preallocated_X,
        preallocated_K,
        preallocated_L,
        preallocated_y,
        preallocated_d,
        observation_noise,
        decision_rule,
        0,
        capacity
    )
end


"""
When the kernel is changed, we need to update d, K, and L
"""
function set_kernel!(s::Surrogate, kernel::RadialBasisFunction)
    @views begin
        N = get_observed(s)
        s.ψ = kernel
        σn2 = get_observation_noise(s)
        s.K[1:N, 1:N] .= eval_KXX(get_kernel(s), get_active_covariates(s)) + (JITTER + σn2) * I
        s.L[1:N, 1:N] .= LowerTriangular(
            cholesky(
                Hermitian(s.K[1:N, 1:N])
            ).L
        )
        s.d[1:N] = s.L[1:N, 1:N]' \ (s.L[1:N, 1:N] \ get_active_observations(s))
    end
end

function set_kernel!(s::HybridSurrogate, kernel::RadialBasisFunction)
    @views begin
        N = get_observed(s)
        s.ψ = kernel
        σn2 = get_observation_noise(s)
        s.K[1:N, 1:N] = eval_KXX(kernel, get_active_covariates(s)) + (JITTER + σn2) * I
        s.L[1:N, 1:N] = LowerTriangular(
            cholesky(
                Hermitian(s.K[1:N, 1:N])
            ).L
        )
        d, λ = coefficient_solve(
            get_active_covariance(s),
            get_active_parametric_basis_matrix(s),
            get_active_observations(s)
        )
        s.d[1:N] = d
        s.λ[:] = λ
    end
end

function set_parametric_component!(s::HybridSurrogate, pbf::PolynomialBasisFunction)
    s.ϕ = pbf
end

set_parametric_component!(s::Surrogate, pbf::PolynomialBasisFunction) = nothing

function set!(s::Surrogate, X::Matrix{T}, y::Vector{T}) where {T<:Real}
    @views begin
        d, N = size(X)

        s.X[:, 1:N] = X
        observation_noise = get_observation_noise(s)
        K = eval_KXX(get_kernel(s), s.X[:, 1:N]) + (JITTER + observation_noise) * I
        s.K[1:N, 1:N] = K
        s.L[1:N, 1:N] = LowerTriangular(
            cholesky(
                Hermitian(
                    s.K[1:N, 1:N]
                )
            ).L
        )
        s.d[1:N] = s.L[1:N, 1:N]' \ (s.L[1:N, 1:N] \ y)
        s.y[1:N] = y
        s.observed = length(y)
    end
end

function set!(s::HybridSurrogate, X::Matrix{T}, y::Vector{T}) where {T<:Real}
    @views begin
        d, N = size(X)

        s.X[:, 1:N] = X
        observation_noise = get_observation_noise(s)
        KXX = eval_KXX(get_kernel(s), s.X[:, 1:N]) + (JITTER + observation_noise) * I
        s.K[1:N, 1:N] = KXX
        s.L[1:N, 1:N] = LowerTriangular(
            cholesky(
                Hermitian(
                    s.K[1:N, 1:N]
                )
            ).L
        )
        ϕ = get_parametric_basis_function(s)
        PX = eval_basis(ϕ, X)
        s.P[1:N, 1:length(ϕ)] = PX
        d, λ = coefficient_solve(KXX, PX, y)
        s.d[1:N] = d
        s.λ[1:length(ϕ)] = λ
        s.y[1:N] = y
        s.observed = length(y)
    end
end


function resize(s::HybridSurrogate)
    return HybridSurrogate(
        get_kernel(s),
        get_parametric_basis_function(s),
        get_covariates(s),
        get_observations(s),
        capacity=get_capacity(s) * DOUBLE,
        decision_rule=get_decision_rule(s),
        observation_noise=get_observation_noise(s)
    )
end


function resize(s::Surrogate)
    return Surrogate(
        get_kernel(s),
        get_covariates(s),
        get_observations(s),
        capacity=get_capacity(s) * DOUBLE,
        decision_rule=get_decision_rule(s)
    )
end


function insert!(s::AbstractSurrogate, x::Vector{T}, y::T) where {T<:Real}
    insert_index = get_observed(s) + 1
    s.X[:, insert_index] = x
    s.y[insert_index] = y
end


function update_covariance!(s::AbstractSurrogate, x::Vector{T}, y::T) where {T<:Real}
    @views begin
        update_index = get_observed(s)
        active_X = get_covariates(s)[:, 1:update_index-1]
        kernel = get_kernel(s)

        # Update the main diagonal
        s.K[update_index, update_index] = kernel(0.0) + get_observation_noise(s) + JITTER
        # Update the rows and columns with covariance vector formed from k(x, X)
        s.K[update_index, 1:update_index-1] = eval_KxX(kernel, x, active_X)'
        s.K[1:update_index-1, update_index] = s.K[update_index, 1:update_index-1]
    end
end

function update_cholesky!(s::AbstractSurrogate)
    # Grab entries from update covariance matrix
    @views begin
        n = get_observed(s)
        B = s.K[n:n, 1:n-1]
        C = s.K[n:n, n:n]
        L = s.L[1:n-1, 1:n-1]

        # Compute the updated factorizations using schur complements
        L21 = B / L'
        L22 = cholesky(C - L21 * L21').L

        # Update the full factorization
        for j in 1:n-1
            s.L[n, j] = L21[1, j]
        end
        s.L[n, n] = L22[1, 1]
    end
end

function update_coefficients!(s::HybridSurrogate)
    update_index = get_observed(s)
    KXX = get_active_covariance(s)
    PX = get_active_parametric_basis_matrix(s)
    y = get_active_observations(s)
    d, λ = coefficient_solve(KXX, PX, y)
    @views begin
        s.d[1:length(d)] = d
        s.λ[:] = λ
    end
end

function update_coefficients!(s::Surrogate)
    update_index = get_observed(s)
    @views begin
        L = s.L[1:update_index, 1:update_index]
        s.d[1:update_index] = L' \ (L \ s.y[1:update_index])
    end
end

function update_parametric_design_matrix!(s::HybridSurrogate)
    @views begin
        update_index = get_observed(s)
        new_x = get_covariates(s)[:, update_index]
        s.P[update_index, :] = eval_basis(get_parametric_basis_function(s), new_x)
    end
end

# Update in place
function condition!(s::HybridSurrogate, xnew::Vector{T}, ynew::T) where {T<:Real}
    if is_full(s)
        s = resize(s)
    end
    insert!(s, xnew, ynew) # Updates covariate matrix X and observation vector y
    increment!(s)
    update_covariance!(s, xnew, ynew) # Updates covariance matrix K
    update_cholesky!(s) # Updates cholesky factorization matrix L
    update_parametric_design_matrix!(s) # Updates parametric design matrix P
    update_coefficients!(s) # Updates coefficients d, λ
    return s
end


function condition!(s::Surrogate, xnew::Vector{T}, ynew::T) where {T<:Real}
    if is_full(s)
        s = resize(s)
    end
    insert!(s, xnew, ynew)
    increment!(s)
    update_covariance!(s, xnew, ynew)
    update_cholesky!(s)
    update_coefficients!(s)
    return s
end


function eval(
    s::HybridSurrogate,
    x::Vector{T},
    θ::Vector{T}) where {T<:Real}
    @views begin
        sx = LazyStruct()
        set(sx, :s, s)
        set(sx, :x, x)
        set(sx, :θ, θ)

        active_index = get_observed(s)
        X = get_active_covariates(s)
        K = get_active_covariance(s)
        P = get_active_parametric_basis_matrix(s)
        L = get_active_cholesky(s)
        d = get_active_coefficients(s)
        λ = get_parametric_component_coefficients(s)
        y = get_active_observations(s)
        kernel = get_kernel(s)
        parametric_basis = get_parametric_basis_function(s)

        dim, N = size(X)
        M = length(λ)

        sx.kx = () -> eval_KxX(kernel, x, X)
        sx.∇kx = () -> eval_∇KxX(kernel, x, X)
        sx.px = () -> eval_basis(parametric_basis, x)
        sx.∇px = () -> eval_∇basis(parametric_basis, x)

        # Predictive mean and its gradient and hessian
        sx.μ = () -> dot(sx.kx, d) + dot(sx.px, λ)
        sx.∇μ = () -> sx.∇kx * d + sx.∇px * λ
        sx.dμ = () -> vcat(sx.μ, sx.∇μ)
        sx.Hμ = function ()
            H = zeros(dim, dim)
            # Reducing over the non-parametric component
            for j = 1:N
                H += d[j] * eval_Hk(kernel, x - X[:, j])
            end
            # Reducing over the parametric component
            HP = eval_Hbasis(parametric_basis, x)
            for j = 1:M
                H += λ[j] * HP[:, :, 1, j]
            end
            return H
        end

        # Reused terms c_i
        sx.c0 = () -> K * sx.w - sx.kx

        # Predictive standard deviation and its gradient and hessian
        sx.A = function ()
            zz = zeros(length(parametric_basis), length(parametric_basis))
            A = [zz P';
                P K]
            return A
        end
        sx.v = () -> [sx.px sx.kx']'
        sx.∇v = () -> [sx.∇px sx.∇kx]
        sx.w = () -> sx.A \ sx.v
        sx.∇w = () -> sx.A \ sx.∇v'
        sx.σ = () -> sqrt(kernel(0) - dot(sx.v, sx.w))
        sx.∇σ = () -> -(sx.∇v * sx.w) / sx.σ
        sx.Hσ = function ()
            H = sx.∇σ * sx.∇σ' + sx.∇v * sx.∇w
            w = sx.w

            HP = eval_Hbasis(parametric_basis, x)
            for j in 1:length(λ)
                H += w[j] * HP[:, :, 1, j]
            end

            for j in length(λ)+1:length(w)
                H += w[j] * eval_Hk(kernel, x - X[:, j-length(λ)])
            end

            H /= -sx.σ

            return H
        end

        sx.y = () -> y
        # Acquisition function and its derivatives, hessians and mixed partials
        sx.g = () -> get_decision_rule(s)

        sx.dg_dμ = () -> first_partial(sx.g, symbol=:μ)(sx.μ, sx.σ, sx.θ, sx)
        sx.dg_dσ = () -> first_partial(sx.g, symbol=:σ)(sx.μ, sx.σ, sx.θ, sx)
        sx.dg_dθ = () -> first_partial(sx.g, symbol=:θ)(sx.μ, sx.σ, sx.θ, sx)

        sx.d2g_dμ = () -> second_partial(sx.s.g, symbol=:μ)(sx.μ, sx.σ, sx.θ, sx)
        sx.d2g_dσ = () -> second_partial(sx.s.g, symbol=:σ)(sx.μ, sx.σ, sx.θ, sx)
        sx.d2g_dθ = () -> second_partial(sx.s.g, symbol=:θ)(sx.μ, sx.σ, sx.θ, sx)

        sx.d2g_dμdθ = () -> mixed_partial(sx.s.g, symbol=:μθ)(sx.μ, sx.σ, sx.θ, sx)
        sx.d2g_dσdθ = () -> mixed_partial(sx.s.g, symbol=:σθ)(sx.μ, sx.σ, sx.θ, sx)

        sx.αxθ = () -> s.g(sx.μ, sx.σ, sx.θ, sx)

        # Spatial derivatives
        sx.∇αx = () -> sx.dg_dμ * sx.∇μ + sx.dg_dσ * sx.∇σ
        sx.Hαx = () -> sx.d2g_dμ * sx.∇μ * sx.∇μ' + sx.dg_dμ * sx.Hμ + sx.d2g_dσ * sx.∇σ * sx.∇σ' + sx.dg_dσ * sx.Hσ

        # Hyperparameter derivatives
        sx.∇αθ = () -> sx.dg_dθ
        sx.Hαθ = () -> sx.d2g_dθ

        # Mixed partials
        sx.d2α_dσdθ = () -> sx.∇σ * sx.d2g_dσdθ'
        sx.d2α_dμdθ = () -> sx.∇μ * sx.d2g_dμdθ'
        sx.d2α_dxdθ = () -> sx.d2α_dμdθ + sx.d2α_dσdθ
    end

    return sx
end


function eval(
    s::Surrogate,
    x::Vector{T},
    θ::Vector{T}) where {T<:Real}
    @views begin
        sx = LazyStruct()
        set(sx, :s, s)
        set(sx, :x, x)
        set(sx, :θ, θ)

        active_index = get_observed(s)
        X = get_active_covariates(s)
        L = get_active_cholesky(s)
        c = get_active_coefficients(s)
        y = get_active_observations(s)
        kernel = get_kernel(s)

        d, N = size(X)

        sx.kx = () -> eval_KxX(kernel, x, X)
        sx.∇kx = () -> eval_∇KxX(kernel, x, X)

        sx.μ = () -> dot(sx.kx, c)
        sx.∇μ = () -> sx.∇kx * c
        sx.dμ = () -> vcat(sx.μ, sx.∇μ)
        sx.Hμ = function ()
            H = zeros(d, d)
            for j = 1:N
                H += c[j] * eval_Hk(kernel, x - X[:, j])
            end
            return H
        end

        sx.w = () -> L' \ (L \ sx.kx)
        sx.Dw = () -> L' \ (L \ (sx.∇kx'))
        sx.∇w = () -> sx.Dw'
        sx.σ = () -> sqrt(kernel(0) - dot(sx.kx', sx.w))
        sx.dσ = function ()
            kxx = eval_Dk(kernel, zeros(d))
            kxX = [eval_KxX(kernel, x, X)'; eval_∇KxX(kernel, x, X)]
            σx = Symmetric(kxx - kxX * (L' \ (L \ kxX')))
            σx = cholesky(σx).L
            return σx
        end
        sx.∇σ = () -> -(sx.∇kx * sx.w) / sx.σ
        sx.Hσ = function ()
            H = -sx.∇σ * sx.∇σ' - sx.∇kx * sx.Dw
            w = sx.w
            for j = 1:N
                H -= w[j] * eval_Hk(kernel, x - X[:, j])
            end
            H /= sx.σ
            return H
        end

        sx.y = () -> y
        sx.g = () -> get_decision_rule(s)

        sx.dg_dμ = () -> first_partial(sx.g, symbol=:μ)(sx.μ, sx.σ, sx.θ, sx)
        sx.dg_dσ = () -> first_partial(sx.g, symbol=:σ)(sx.μ, sx.σ, sx.θ, sx)
        sx.dg_dθ = () -> first_partial(sx.g, symbol=:θ)(sx.μ, sx.σ, sx.θ, sx)

        sx.d2g_dμ = () -> second_partial(sx.s.g, symbol=:μ)(sx.μ, sx.σ, sx.θ, sx)
        sx.d2g_dσ = () -> second_partial(sx.s.g, symbol=:σ)(sx.μ, sx.σ, sx.θ, sx)
        sx.d2g_dθ = () -> second_partial(sx.s.g, symbol=:θ)(sx.μ, sx.σ, sx.θ, sx)

        sx.d2g_dμdθ = () -> mixed_partial(sx.s.g, symbol=:μθ)(sx.μ, sx.σ, sx.θ, sx)
        sx.d2g_dσdθ = () -> mixed_partial(sx.s.g, symbol=:σθ)(sx.μ, sx.σ, sx.θ, sx)

        sx.αxθ = () -> s.g(sx.μ, sx.σ, sx.θ, sx)

        # Spatial derivatives
        sx.∇αx = () -> sx.dg_dμ * sx.∇μ + sx.dg_dσ * sx.∇σ
        sx.Hαx = () -> sx.d2g_dμ * sx.∇μ * sx.∇μ' + sx.dg_dμ * sx.Hμ + sx.d2g_dσ * sx.∇σ * sx.∇σ' + sx.dg_dσ * sx.Hσ

        # Hyperparameter derivatives
        sx.∇αθ = () -> sx.dg_dθ
        sx.Hαθ = () -> sx.d2g_dθ

        # Mixed partials
        sx.d2α_dσdθ = () -> sx.∇σ * sx.d2g_dσdθ'
        sx.d2α_dμdθ = () -> sx.∇μ * sx.d2g_dμdθ'
        sx.d2α_dxdθ = () -> sx.d2α_dμdθ + sx.d2α_dσdθ
    end

    return sx
end

(s::AbstractSurrogate)(x, θ) = eval(s, x, θ)
eval(sx) = sx.αxθ
gradient(sx; wrt_hypers=false) = wrt_hypers ? sx.∇αθ : sx.∇αx
hessian(sx; wrt_hypers=false) = wrt_hypers ? sx.Hαθ : sx.Hαx
mixed_partials(sx) = sx.d2α_dxdθ


# ------------------------------------------------------------------
# Operations for computing optimal hyperparameters.
# ------------------------------------------------------------------
function log_likelihood(s::HybridSurrogate)
    n = get_observed(s)
    m = length(get_parametric_basis_function(s))
    yz = [get_active_observations(s); zeros(m)]
    d = get_active_coefficients(s)
    λ = get_parametric_component_coefficients(s)
    dλ = [d; λ]
    P = get_active_parametric_basis_matrix(s)
    K = get_active_covariance(s)

    M = [zeros(m, m) P';
        P K]
    ladM = first(logabsdet(M))

    return -dot(yz, dλ) / 2 - n * log(2π) / 2 - ladM
end

function δlog_likelihood(s::HybridSurrogate, δθ::Vector{T}) where {T<:Real}
    kernel = get_kernel(s)
    X = get_active_covariates(s)
    pdim = length(get_parametric_basis_function(s))
    kdim = size(X, 2)
    zz = zeros(pdim, pdim)
    zkp = zeros(kdim, pdim)

    δK = eval_Dθ_KXX(kernel, X, δθ)
    δKhat = [zz zkp';
        zkp δK]
    d = get_active_coefficients(s)
    λ = get_parametric_component_coefficients(s)
    # dλ = [d; λ]
    dλ = [λ; d]
    K = get_active_cholesky(s)
    P = get_active_parametric_basis_matrix(s)

    Khat = [zz P';
        P K]
    return (dλ' * δKhat * dλ - tr(Khat \ δKhat)) / 2
end

function log_likelihood(s::Surrogate)
    n = get_observed(s)
    y = get_active_observations(s)
    c = get_active_coefficients(s)
    L = get_active_cholesky(s)
    return -y' * c / 2 - sum(log.(diag(L))) - n * log(2π) / 2
end

function δlog_likelihood(s::Surrogate, δθ::Vector{T}) where {T<:Real}
    kernel = get_kernel(s)
    X = get_active_covariates(s)
    δK = eval_Dθ_KXX(kernel, X, δθ)
    c = get_active_coefficients(s)
    L = get_active_cholesky(s)
    return (c' * δK * c - tr(L' \ (L \ δK))) / 2
end

function ∇log_likelihood(s::AbstractSurrogate)
    nθ = length(s.ψ.θ)
    δθ = zeros(nθ)
    ∇L = zeros(nθ)

    for j in 1:nθ
        δθ[:] .= 0.0
        δθ[j] = 1.0
        ∇L[j] = δlog_likelihood(s, δθ)
    end

    return ∇L
end

"""
This only optimizes for lengthscale hyperparameter where the lengthscale is the
same in each respective dimension.
"""
function optimize!(
    s::AbstractSurrogate;
    starts::Matrix{T},
    lowerbounds::Vector{T},
    upperbounds::Vector{T}) where {T<:Real}
    candidates = []

    for i in 1:size(starts, 2)
        push!(
            candidates,
            hyperparameter_solve(
                s,
                start=starts[:, i],
                lowerbounds=lowerbounds,
                upperbounds=upperbounds
            )
        )
    end

    candidates = filter(pair -> !any(isnan.(pair[1])), candidates)
    mini, j_mini = findmin(pair -> pair[2], candidates)
    θ = candidates[j_mini][1]
    set_kernel!(s, set_hyperparameters!(get_kernel(s), θ))

    return nothing
end

# function hyperparameter_solve(
#     s::Surrogate;
#     start,
#     lowerbounds,
#     upperbounds,
#     optim_options
# )

#     function fg!(F, G, θ::Vector{T}) where {T<:Real}
#         set_kernel!(s, set_hyperparameters!(get_kernel(s), θ))
#         if G !== nothing
#             G .= -∇log_likelihood(s)
#         end
#         if F !== nothing
#             return -log_likelihood(s)
#         end
#     end

#     res = optimize(
#         Optim.only_fg!(fg!),
#         lowerbounds,
#         upperbounds,
#         start,
#         Fminbox(LBFGS()),
#         optim_options
#     )

#     return (Optim.minimizer(res), Optim.minimum(res))
# end

function log_likelihood_constructor(s::AbstractSurrogate)
    θ = θ -> begin
        # Note: this updates the given surrogate
        set_kernel!(s, set_hyperparameters!(get_kernel(s), θ))
        return -log_likelihood(s)
    end

    ∇θ = θ -> begin
        set_kernel!(s, set_hyperparameters!(get_kernel(s), θ))
        return -∇log_likelihood(s)
    end

    return (θ, ∇θ)
end

function projected_gradient_descent(f, g, x0;
    lower=[-1.0],
    upper=[1.0],
    maxiter=50,
    tol=1e-8,
    α=0.1)
    # Project x onto [lower, upper]
    clamp_in_bounds(x) = clamp(x, lower, upper)

    x = clamp_in_bounds(x0)
    for iter in 1:maxiter
        grad = g(x)

        # Check for convergence based on small gradient
        if norm(grad) < tol
            # println("Converged at iteration $iter")
            return x
        end

        # Take a gradient step
        x_proposed = x - α * grad
        # Project the result back into the feasible region
        x_new = clamp_in_bounds(x_proposed)

        # If the update is tiny, consider it converged
        if norm(x_new - x) < tol
            # println("Converged at iteration $iter (step size small)")
            return x_new
        end

        x = x_new
    end

    # println("Warning: Reached maxiter = $maxiter without convergence")
    return x
end

function hyperparameter_solve(
    s::AbstractSurrogate;
    start,
    lowerbounds,
    upperbounds)
    f, g = log_likelihood_constructor(s)
    minimizer = projected_gradient_descent(f, g, start; lower=lowerbounds, upper=upperbounds)

    return (minimizer, f(minimizer))
end

Distributions.mean(sx) = sx.μ
Distributions.std(sx) = sqrt(sx.σ)