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
get_dim(s::AbstractSurrogate) = size(s.X, 1)


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
    # L::Matrix{Float64}
    y::Vector{Float64}
    d::Vector{Float64} # Coefficients for Gaussian process
    λ::Vector{Float64} # Coefficients for parametric/trend term
    observation_noise::Float64
    g::DecisionRule
    observed::Int
    capacity::Int
    containers::PreallocatedContainers
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
    # L::Matrix{Float64}
    y::Vector{Float64}
    d::Vector{Float64}
    observation_noise::Float64
    g::DecisionRule
    observed::Int
    capacity::Int
    containers::PreallocatedContainers
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
    y::Vector{T},
    capacity::Int=DEFAULT_CAPACITY,
    decision_rule::AbstractDecisionRule=EI(),
    observation_noise::T=1e-6) where {T<:Real}
    @assert length(y) <= capacity "Capacity must be >= number of observations."
    dim, N = size(X)

    """
    Preallocate a matrix for covariates of size d x capacity where capacity is the maximum
    number of observations.
    """
    preallocated_X = zeros(dim, capacity)
    preallocated_X[:, 1:N] = X

    """
    Preallocate a matrix for the matrix that represents the polynomial basis evaluation of 
    our observations of size capacity x m where m is the dimensionality of our basis vector.
    """
    preallocated_P = zeros(capacity, length(ϕ))
    # PX = eval_basis(ϕ, X)
    eval_basis!(ϕ, X, (@view preallocated_P[1:N, 1:length(ϕ)]))
    PX = preallocated_P[1:N, 1:length(ϕ)]
    # preallocated_P[1:N, 1:length(ϕ)] = PX

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
    # preallocated_L = zeros(capacity, capacity)
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

    containers = PreallocatedContainers(length(ϕ), dim, capacity, length(ψ))

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
        capacity,
        containers
    )
end

# Constructor that expects observations and covariates later
function HybridSurrogate(
    ψ::RadialBasisFunction,
    ϕ::PolynomialBasisFunction,
    dim::Int,
    capacity::Int=DEFAULT_CAPACITY,
    decision_rule::AbstractDecisionRule=EI(),
    observation_noise::T=1e-6) where {T<:Real}
    preallocated_X = zeros(dim, capacity)
    preallocated_P = zeros(capacity, length(ϕ))
    preallocated_K = zeros(capacity, capacity)
    preallocated_L = LowerTriangular(zeros(capacity, capacity))
    # preallocated_L = zeros(capacity, capacity)
    preallocated_d = zeros(capacity)
    λ_polynomial = zeros(length(ϕ))
    preallocated_y = zeros(capacity)
    observed = 0
    containers = PreallocatedContainers(length(ϕ), dim, capacity, length(ψ))

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
        capacity,
        containers
    )
end

function Surrogate(
    ψ::RadialBasisFunction,
    X::Matrix{T},
    y::Vector{T},
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
    # preallocated_L = zeros(capacity, capacity)
    preallocated_L[1:N, 1:N] = LowerTriangular(
        cholesky(
            Hermitian(
                preallocated_K[1:N, 1:N]
            )
        ).L
    )

    preallocated_d = zeros(capacity)
    preallocated_d[1:N] = preallocated_L[1:N, 1:N]' \ (preallocated_L[1:N, 1:N] \ y)

    preallocated_y = zeros(capacity)
    preallocated_y[1:N] = y

    containers = PreallocatedContainers(1, d, capacity, length(ψ))

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
        capacity,
        containers
    )
end

function Surrogate(
    ψ::RadialBasisFunction,
    dim::Int,
    capacity::Int=DEFAULT_CAPACITY,
    decision_rule::AbstractDecisionRule=EI(),
    observation_noise::T=1e-6) where {T<:Real}
    preallocated_X = zeros(dim, capacity)
    preallocated_K = zeros(capacity, capacity)
    preallocated_L = LowerTriangular(zeros(capacity, capacity))
    # preallocated_L = zeros(capacity, capacity)
    preallocated_d = zeros(capacity)
    preallocated_y = zeros(capacity)

    containers = PreallocatedContainers(1, dim, capacity, length(ψ))

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
        capacity,
        containers
    )
end


get_active_KxX(s::AbstractSurrogate) = @view s.containers.KxX[1:get_observed(s)]
get_active_grad_KxX(s::AbstractSurrogate) = @view s.containers.grad_KxX[:, 1:get_observed(s)]
get_active_δKXX(s::AbstractSurrogate) = @view s.containers.δKXX[1:get_observed(s), 1:get_observed(s)]
get_active_Hk(s::AbstractSurrogate) = @view s.containers.Hk[:, :]
get_active_Hσ(s::AbstractSurrogate) = @view s.containers.Hσ[:, :]


"""
When the kernel is changed, we need to update d, K, and L
"""
function set_kernel!(s::Surrogate, kernel::RadialBasisFunction)
    @views begin
        N = get_observed(s)
        s.ψ = kernel
        observation_noise = get_observation_noise(s)
        eval_KXX!(
            kernel,
            get_active_covariates(s),
            get_active_covariance(s),
            s.containers.diff_x
        )
        s.K[1:N, 1:N] += (JITTER + observation_noise) * I
        s.L[1:N, 1:N] = LowerTriangular(
            cholesky(
                Hermitian(get_active_covariance(s))
            ).L
        )
        s.d[1:N] = s.L[1:N, 1:N]' \ (s.L[1:N, 1:N] \ get_active_observations(s))
    end
end

function set_kernel!(s::HybridSurrogate, kernel::RadialBasisFunction)
    diff_x = zeros(get_dim(s))
    @views begin
        N = get_observed(s)
        s.ψ = kernel
        observation_noise = get_observation_noise(s)
        # s.K[1:N, 1:N] = eval_KXX(kernel, get_active_covariates(s)) + (JITTER + σn2) * I
        eval_KXX!(
            kernel,
            get_active_covariates(s),
            get_active_covariance(s),
            s.containers.diff_x
        )
        s.K[1:N, 1:N] += (JITTER + observation_noise) * I
        s.L[1:N, 1:N] = LowerTriangular(
            cholesky(
                Hermitian(get_active_covariance(s))
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
        dim, N = size(X)

        s.X[:, 1:N] = X
        s.observed = N
        observation_noise = get_observation_noise(s)
        eval_KXX!(
            get_kernel(s),
            get_active_covariates(s),
            get_active_covariance(s),
            s.containers.diff_x
        )
        s.K[1:N, 1:N] += (JITTER + observation_noise) * I
        s.L[1:N, 1:N] = LowerTriangular(
            cholesky(
                Hermitian(
                    get_active_covariance(s)
                )
            ).L
        )
        s.d[1:N] = s.L[1:N, 1:N]' \ (s.L[1:N, 1:N] \ y)
        s.y[1:N] = y
    end
end

function set!(s::HybridSurrogate, X::Matrix{T}, y::Vector{T}) where {T<:Real}
    @views begin
        dim, N = size(X)

        s.X[:, 1:N] = X
        s.observed = N
        observation_noise = get_observation_noise(s)
        eval_KXX!(
            get_kernel(s),
            get_active_covariates(s),
            get_active_covariance(s),
            s.containers.diff_x
        )
        s.K[1:N, 1:N] += (JITTER + observation_noise) * I
        s.L[1:N, 1:N] = LowerTriangular(
            cholesky(
                Hermitian(
                    get_active_covariance(s)
                )
            ).L
        )
        eval_basis!(
            get_parametric_basis_function(s),
            get_active_covariates(s),
            get_active_parametric_basis_matrix(s)
        )
            # s.P[1:N, 1:length(ϕ)])
        # PX = s.P[1:N, 1:length(ϕ)]
        # s.P[1:N, 1:length(ϕ)] = PX
        d, λ = coefficient_solve(
            get_active_covariance(s),
            get_active_parametric_basis_matrix(s),
            y
        )
        s.d[1:N] = d
        s.λ[1:length(ϕ)] = λ
        s.y[1:N] = y
    end
end


function resize(s::HybridSurrogate)
    return HybridSurrogate(
        get_kernel(s),
        get_parametric_basis_function(s),
        get_covariates(s),
        get_observations(s),
        get_capacity(s) * DOUBLE,
        get_decision_rule(s),
        get_observation_noise(s)
    )
end


function resize(s::Surrogate)
    return Surrogate(
        get_kernel(s),
        get_covariates(s),
        get_observations(s),
        get_capacity(s) * DOUBLE,
        get_decision_rule(s),
        get_observation_noise(s)
    )
end


function insert!(s::AbstractSurrogate, x::Vector{T}, y::T) where {T<:Real}
    insert_index = get_observed(s) + 1
    s.X[:, insert_index] = x
    s.y[insert_index] = y
end


function update_covariance!(s::AbstractSurrogate, x::Vector{T}) where {T<:Real}
    @views begin
        update_index = get_observed(s)
        active_X = get_covariates(s)[:, 1:update_index-1]
        kernel = get_kernel(s)

        # Update the main diagonal
        s.K[update_index, update_index] = kernel(0.0) + get_observation_noise(s) + JITTER
        # Update the rows and columns with covariance vector formed from k(x, X)
        # s.K[update_index, 1:update_index-1] = eval_KxX(kernel, x, active_X)'
        eval_KxX!(
            kernel,
            x,
            active_X,
            s.K[update_index, 1:update_index-1],
            s.containers.KxX
        )
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
        # s.P[update_index, :] = eval_basis(get_parametric_basis_function(s), new_x)
        eval_basis!(
            get_parametric_basis_function(s),
            new_x,
            s.P[update_index:update_index, :]
        )
    end
end

# Update in place
function condition!(s::HybridSurrogate, xnew::Vector{T}, ynew::T) where {T<:Real}
    if is_full(s)
        s = resize(s)
    end
    insert!(s, xnew, ynew) # Updates covariate matrix X and observation vector y
    increment!(s)
    update_covariance!(s, xnew) # Updates covariance matrix K
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
    update_covariance!(s, xnew)
    update_cholesky!(s)
    update_coefficients!(s)
    return s
end


function eval(
    s::HybridSurrogate,
    x::AbstractVector{T},
    θ::AbstractVector{T}) where {T<:Real}
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
        sx.px = function () 
            eval_basis!(parametric_basis, x, s.containers.px)
            return s.containers.px
        end
        sx.∇px = function ()
            eval_∇basis!(parametric_basis, x, s.containers.∇px)
            return s.containers.∇px
        end

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
    x::AbstractVector{T},
    θ::AbstractVector{T}) where {T<:Real}
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

        sx.kx = function()
            return eval_KxX!(
                kernel,
                x,
                X,
                get_active_KxX(sx.s),
                sx.s.containers.diff_x
            )
        end
        sx.∇kx = function() 
            return eval_∇KxX!(
                kernel,
                x,
                X,
                get_active_grad_KxX(sx.s),
                sx.s.containers.diff_x
            )
        end

        sx.μ = () -> dot(sx.kx, c)
        sx.∇μ = () -> sx.∇kx * c
        sx.dμ = () -> vcat(sx.μ, sx.∇μ)
        sx.Hμ = function ()
            H = get_active_Hk(sx.s)
            Hf = zeros(size(H))
            fill!(H, 0.)
            for j = 1:N
                Hf += c[j] * eval_Hk!(
                    kernel,
                    x - X[:, j],
                    H
                )
            end
            return Hf
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
            Hk = get_active_Hk(sx.s)
            Hσ = get_active_Hσ(sx.s)
            fill!(Hk, 0.)
            fill!(Hσ, 0.)
            Hσ .= -sx.∇σ * sx.∇σ' - sx.∇kx * sx.Dw
            w = sx.w
            for j = 1:N
                Hσ .-= w[j] * eval_Hk!(kernel, x - X[:, j], Hk)
            end
            Hσ /= sx.σ
            return Hσ
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
(s::AbstractSurrogate)(x, θ, sx) = eval(s, x, θ, sx)
function eval(sx::LazyStruct)::Float64
    return sx.αxθ
end
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

    M = Matrix{Float64}([zeros(m, m) P';
        P K])
    ladM = log(abs(det(M)))

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
    δK = eval_Dθ_KXX!(
        get_kernel(s),
        get_active_covariates(s),
        δθ,
        get_active_δKXX(s),
        s.containers.diff_x,
        s.containers.δψij
    )
    c = get_active_coefficients(s)
    L = get_active_cholesky(s)
    return (c' * δK * c - tr(L' \ (L \ δK))) / 2
end

function ∇log_likelihood(s::AbstractSurrogate)
    nθ = length(s.ψ.θ)

    for j in 1:nθ
        s.containers.δθ[:] .= 0.0
        s.containers.δθ[j] = 1.0
        s.containers.grad_L[j] = δlog_likelihood(s, s.containers.δθ)
    end

    return s.containers.grad_L
end


function optimize!(
    # s::AbstractSurrogate;
    s::Surrogate;
    lowerbounds::Vector{T},
    upperbounds::Vector{T},
    starts::Matrix{T},
    minimizers_container::Vector{Vector{T}},
    minimums_container::Vector{T},
    optim_options = Optim.Options(
        outer_iterations=100,
        x_tol=1e-3,
        f_tol=1e-3,
        time_limit=NEWTON_SOLVE_TIME_LIMIT
    )) where T <: Real

    function fg!(F, G, θ::Vector{T}) where T <: Real
        set_kernel!(s, set_hyperparameters!(get_kernel(s), θ))
        # TODO: log likelihood is allocating. Fix this here
        if G !== nothing G .= -∇log_likelihood(s) end
        if F !== nothing return -log_likelihood(s) end
    end

    M = size(starts, 2)
    for i in 1:M
        res = optimize(
            Optim.only_fg!(fg!),
            lowerbounds,
            upperbounds,
            starts[:, i],
            Fminbox(LBFGS()),
            optim_options
        )
        minimizers_container[i] = Optim.minimizer(res)
        minimums_container[i] = Optim.minimum(res)
    end

    candidates = [(minimizers_container[i], minimums_container[i]) for i in 1:M]
    candidates = filter(pair -> !any(isnan.(pair[1])), candidates)
    mini, j_mini = findmin(pair -> pair[2], candidates)
    # println("Candidate: $(candidates[j_mini])")
    θ = candidates[j_mini][1]
    set_kernel!(s, set_hyperparameters!(get_kernel(s), θ))

    return nothing
end

Distributions.mean(sx) = sx.μ
Distributions.std(sx) = sqrt(sx.σ)