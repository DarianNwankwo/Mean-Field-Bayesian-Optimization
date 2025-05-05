# Useful functions for grabbing desired attributes on our structs
get_kernel(afs::AbstractSurrogate) = afs.ψ
get_covariates(afs::AbstractSurrogate) = afs.X
get_observations(afs::AbstractSurrogate) = afs.y
get_coefficients(afs::AbstractSurrogate) = afs.d
get_cholesky(as::AbstractSurrogate) = as.L
get_covariance(as::AbstractSurrogate) = as.K
get_covariance_scratchpad(as::AbstractSurrogate) = as.Kscratch
get_capacity(s::AbstractSurrogate) = s.capacity
increment!(s::AbstractSurrogate) = s.observed += 1
get_observed(s::AbstractSurrogate) = s.observed
is_full(s::AbstractSurrogate) = get_observed(s) == get_capacity(s)
get_active_covariates(s::AbstractSurrogate) = @view get_covariates(s)[:, 1:get_observed(s)]
get_active_cholesky(s::AbstractSurrogate) = @view get_cholesky(s)[1:get_observed(s), 1:get_observed(s)]
get_active_covariance(s::AbstractSurrogate) = @view get_covariance(s)[1:get_observed(s), 1:get_observed(s)]
get_active_covariance_scratchpad(s::AbstractSurrogate) = @view get_covariance_scratchpad(s)[1:get_observed(s), 1:get_observed(s)]
get_active_observations(s::AbstractSurrogate) = @view get_observations(s)[1:get_observed(s)]
get_active_coefficients(s::AbstractSurrogate) = @view get_coefficients(s)[1:get_observed(s)]
get_observation_noise(s::AbstractSurrogate) = s.observation_noise
get_dim(s::AbstractSurrogate) = size(s.X, 1)
get_schur_buffer(s::AbstractSurrogate) = s.containers.schur


function schur_reduced_system_solve(
    L::AbstractMatrix{<:Real},
    PX::AbstractMatrix{<:Real},
    Px::AbstractMatrix{<:Real},
    KxX::AbstractVector{<:Real},
    schur::SchurBuffer)
    # Preallocate scratch buffers
    n, m = size(PX)
    Y = get_Y(schur)
    S = get_S(schur)
    tmp = get_tmp(schur)
    r = get_r(schur)

    # term1: Y = L \ PX
    @timeit to "schur solve term0" begin
        copyto!(Y, PX)
    end
    
    @timeit to "schur solve term1" begin
        ldiv!(LowerTriangular(L), Y)
    end

    # term2: S = Y' * Y
    @timeit to "schur solve term2" mul!(S, transpose(Y), Y)

    # term3: r = Y'*(L \ KxX) - Px
    @timeit to "schur solve term3" begin
        copyto!(tmp, KxX)
        ldiv!(LowerTriangular(L), tmp)
        mul!(r, transpose(Y), tmp)
        r .-= Px
    end

    # term4: Cholesky factor of S
    @timeit to "schur solve term4" fS = cholesky!(Hermitian(S))

    # term5: w1 = fS \ r  (in-place solve)
    w1 = get_w1(schur)
    @timeit to "schur solve term5" begin
        copyto!(w1, r)
        ldiv!(fS, w1)
    end

    # term6: w0 = L'\(L\(KxX - PX*w1))
    w0 = get_w0(schur)
    @timeit to "schur solve term6" begin
        copyto!(w0, KxX)
        mul!(w0, PX, w1, -1.0, 1.0)     # w0 = KxX - PX*w1
        ldiv!(LowerTriangular(L), w0)
        ldiv!(UpperTriangular(L'), w0)
    end

    # term7: return concatenated solution
    w = get_w(schur)
    w[1:length(w1)] .= w1
    w[length(w1)+1:length(w1) + schur.active_index] .= w0
    @timeit to "schur solve term7" return w
end

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
mutable struct HybridSurrogate{
    RBF <: StationaryKernel,
    PBF <: ParametricRepresentation,
    PM <: ElasticPDMat} <: AbstractSurrogate
    ψ::RBF
    ϕ::PBF
    X::Matrix{Float64} # Covariates
    P::Matrix{Float64} # Parametric term design matrix
    K::Matrix{Float64} # Covariance matrix for Gaussian process
    Kscratch::Matrix{Float64}
    L::PM # Cholesky factorization of covariance matrix
    y::Vector{Float64}
    d::Vector{Float64} # Coefficients for Gaussian process
    λ::Vector{Float64} # Coefficients for parametric/trend term
    observation_noise::Float64
    observed::Int
    capacity::Int
    containers::PreallocatedContainers
end

get_parametric_basis_matrix(as::HybridSurrogate) = as.P
get_active_parametric_basis_matrix(s::HybridSurrogate) = @view get_parametric_basis_matrix(s)[1:get_observed(s), :]
get_parametric_basis_function(s::HybridSurrogate) = s.ϕ
get_parametric_component_coefficients(afs::HybridSurrogate) = afs.λ


# Define the custom show method for Surrogate
function Base.show(io::IO, s::HybridSurrogate{RBF}) where {RBF}
    print(io, "HybridSurrogate{RBF = ")
    show(io, s.ψ)    # Use the show method for RBF
    print(io, "}")
end


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
# function coefficient_solve(
#     KXX::AbstractMatrix{T1},
#     PX::AbstractMatrix{T2},
#     y::AbstractVector{T3}) where {T1<:Real,T2<:Real,T3<:Real}
#     p_dim, k_dim = size(PX, 2), size(KXX, 2)
#     A = [KXX PX;
#         PX' zeros(p_dim, p_dim)]
#     F = lu(A)
#     b = [y; zeros(p_dim)]
#     w = F \ b
#     d, λ = w[1:k_dim], w[k_dim+1:end]
#     return d, λ
# end

function coefficient_solve(
    L::AbstractMatrix{<:Real},
    PX::AbstractMatrix{<:Real},
    fx::AbstractVector{<:Real},
    schur::SchurBuffer)
    @timeit to "get buffers" begin
    d = get_d(schur)
    λ = get_λ(schur)
    Y = get_Y(schur)
    S = get_S(schur)
    r = get_r(schur)
    tmp = get_tmp(schur)
    end


    @timeit to "copyto" copyto!(Y, PX)
    # Y = L \ PX
    # TODO Can insert ElasticPDMat here
    @timeit to "ldvi LY" ldiv!(LowerTriangular(L), Y)
    # S = Y' * Y
    @timeit to "S" mul!(S, transpose(Y), Y)

    # r = PX' * (L' \ (L \ fx)), all in-place
    @timeit to "fx copy" copyto!(tmp, fx)
    @timeit to "ldiv tmp1" ldiv!(LowerTriangular(L), tmp)
    @timeit to "ldiv tmp2" ldiv!(UpperTriangular(L'), tmp)

    # Solve S * λ = r in-place
    @timeit to "mul S" mul!(r, transpose(PX), tmp)
    @timeit to "cholesky S" fS = cholesky!(Hermitian(S))
    @timeit to "copy r" copyto!(λ, r)
    @timeit to "ldiv lambda" ldiv!(fS, λ)

    # 5) Solve for d = L' \ (L \ (fx - PX*λ)) in-place
    @timeit to "fx copy" copyto!(tmp, fx)
    @timeit to "mul px lambda" mul!(tmp, PX, λ, -1.0, 1.0)
    @timeit to "ldiv tmp3" ldiv!(LowerTriangular(L), tmp)
    @timeit to "ldiv tmp4" ldiv!(UpperTriangular(L'), tmp)
    @timeit to "copy last" copyto!(d, tmp)

    @timeit to "return" return d, λ
end

function coefficient_solve2(
    L::ElasticPDMat,
    PX::AbstractMatrix{<:Real},
    fx::AbstractVector{<:Real},
    schur::SchurBuffer)
    @timeit to "get buffers" begin
    d = get_d(schur)
    λ = get_λ(schur)
    Y = get_Y(schur)
    S = get_S(schur)
    r = get_r(schur)
    tmp = get_tmp(schur)
    end


    @timeit to "copyto" copyto!(Y, PX)
    # Y = L \ PX
    # TODO Can insert ElasticPDMat here
    @timeit to "ldvi LY" ldiv!(LowerTriangular(L), Y)
    # S = Y' * Y
    @timeit to "S" mul!(S, transpose(Y), Y)

    # r = PX' * (L' \ (L \ fx)), all in-place
    @timeit to "fx copy" copyto!(tmp, fx)
    @timeit to "ldiv tmp1" ldiv!(LowerTriangular(L), tmp)
    @timeit to "ldiv tmp2" ldiv!(UpperTriangular(L'), tmp)

    # Solve S * λ = r in-place
    @timeit to "mul S" mul!(r, transpose(PX), tmp)
    @timeit to "cholesky S" fS = cholesky!(Hermitian(S))
    @timeit to "copy r" copyto!(λ, r)
    @timeit to "ldiv lambda" ldiv!(fS, λ)

    # 5) Solve for d = L' \ (L \ (fx - PX*λ)) in-place
    @timeit to "fx copy" copyto!(tmp, fx)
    @timeit to "mul px lambda" mul!(tmp, PX, λ, -1.0, 1.0)
    @timeit to "ldiv tmp3" ldiv!(LowerTriangular(L), tmp)
    @timeit to "ldiv tmp4" ldiv!(UpperTriangular(L'), tmp)
    @timeit to "copy last" copyto!(d, tmp)

    @timeit to "return" return d, λ
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
    observation_noise::T=1e-6) where {T<:Real}
    @assert length(y) <= capacity "Capacity must be >= number of observations."
    dim, N = size(X)
    containers = PreallocatedContainers{T}(length(ϕ), dim, capacity, length(ψ), N)

    """
    Preallocate a matrix for covariates of size d x capacity where capacity is the maximum
    number of observations.
    """
    preallocated_X = zeros(T, dim, capacity)
    preallocated_X[:, 1:N] = X

    """
    Preallocate a matrix for the matrix that represents the polynomial basis evaluation of 
    our observations of size capacity x m where m is the dimensionality of our basis vector.
    """
    preallocated_P = zeros(T, capacity, length(ϕ))
    eval_basis!(ϕ, X, (@view preallocated_P[1:N, 1:length(ϕ)]))
    PX = @view preallocated_P[1:N, 1:length(ϕ)]

    """
    Preallocate a covariance matrix of size d x capacity
    """
    preallocated_K = zeros(T, capacity, capacity)
    preallocated_Kscratch = zeros(T, capacity, capacity)
    eval_KXX!(ψ, X, (@view preallocated_Kscratch[1:N, 1:N]), containers.diff_x)
    eval_KXX!(ψ, X, (@view preallocated_K[1:N, 1:N]), containers.diff_x)
    for jj in 1:N
        preallocated_Kscratch[jj, jj] +=  (JITTER + observation_noise)
        preallocated_K[jj, jj] +=  (JITTER + observation_noise)
    end

    """
    Preallocate a matrix for the cholesky factorization of size d x capacity
    """
    # preallocated_L = LowerTriangular(zeros(T, capacity, capacity))
    # preallocated_L[1:N, 1:N] = cholesky(
    #     Hermitian(
    #         preallocated_K[1:N, 1:N]
    #     )
    # ).L
    preallocated_L = ElasticPDMat((@view preallocated_Kscratch[1:N, 1:N]), capacity=capacity)

    """
    Linear system solve for learning coefficients of stochastic component and parametric
    component. 
    """
    d, λ = coefficient_solve(
        view(preallocated_L.chol).L,
        PX,
        y,
        containers.schur
    )

    preallocated_d = zeros(T, capacity)
    preallocated_d[1:length(d)] = d

    λ_polynomial = zeros(T, length(λ))
    λ_polynomial[:] = λ

    preallocated_y = zeros(T, capacity)
    preallocated_y[1:N] = y

    observed = length(y)

    return HybridSurrogate(
        ψ,
        ϕ,
        preallocated_X,
        preallocated_P,
        preallocated_K,
        preallocated_Kscratch,
        preallocated_L,
        preallocated_y,
        preallocated_d,
        λ_polynomial,
        observation_noise,
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
    observation_noise::T=1e-6) where {T<:Real}
    preallocated_X = zeros(T, dim, capacity)
    preallocated_P = zeros(T, capacity, length(ϕ))
    preallocated_K = zeros(T, capacity, capacity)
    preallocated_Kscratch = zeros(T, capacity, capacity)
    # preallocated_L = LowerTriangular(zeros(T, capacity, capacity))
    preallocated_L = ElasticPDMat(capacity=capacity)
    preallocated_d = zeros(T, capacity)
    λ_polynomial = zeros(T, length(ϕ))
    preallocated_y = zeros(T, capacity)
    observed = 0
    containers = PreallocatedContainers{T}(length(ϕ), dim, capacity, length(ψ), 0)

    return HybridSurrogate(
        ψ,
        ϕ,
        preallocated_X,
        preallocated_P,
        preallocated_K,
        preallocated_Kscratch,
        preallocated_L,
        preallocated_y,
        preallocated_d,
        λ_polynomial,
        observation_noise,
        observed,
        capacity,
        containers
    )
end

mutable struct Surrogate{RBF<:StationaryKernel, PM <: ElasticPDMat} <: AbstractSurrogate
    ψ::RBF
    X::Matrix{Float64}
    K::Matrix{Float64}
    Kscratch::Matrix{Float64}
    L::PM
    y::Vector{Float64}
    d::Vector{Float64}
    observation_noise::Float64
    observed::Int
    capacity::Int
    containers::PreallocatedContainers
end

function Surrogate(
    ψ::RadialBasisFunction,
    X::Matrix{T},
    y::Vector{T},
    capacity::Int=DEFAULT_CAPACITY,
    observation_noise::T=1e-6) where T <: Real
    @assert length(y) <= capacity "Capacity must be >= number of observations."
    d, N = size(X)
    UNUSED_NUMBER_OF_BASIS_FUNCTIONS = 1
    containers = PreallocatedContainers{T}(
        UNUSED_NUMBER_OF_BASIS_FUNCTIONS,
        d,
        capacity,
        length(ψ),
        N
    )

    preallocated_X = zeros(T, d, capacity)
    preallocated_X[:, 1:N] = X

    preallocated_K = zeros(T, capacity, capacity)
    preallocated_Kscratch = zeros(T, capacity, capacity)
    eval_KXX!(ψ, X, (@view preallocated_Kscratch[1:N, 1:N]), containers.diff_x)
    eval_KXX!(ψ, X, (@view preallocated_K[1:N, 1:N]), containers.diff_x)
    for jj in 1:N
        preallocated_Kscratch[jj, jj] +=  (JITTER + observation_noise)
        preallocated_K[jj, jj] +=  (JITTER + observation_noise)
    end

    preallocated_L = ElasticPDMat((@view preallocated_Kscratch[1:N, 1:N]), capacity=capacity)
    preallocated_d = zeros(T, capacity)
    # preallocated_d[1:N] = preallocated_L[1:N, 1:N]' \ (preallocated_L[1:N, 1:N] \ y)
    preallocated_d[1:N] = preallocated_L \ y

    preallocated_y = zeros(T, capacity)
    preallocated_y[1:N] = y

    return Surrogate(
        ψ,
        preallocated_X,
        preallocated_K,
        preallocated_Kscratch,
        preallocated_L,
        preallocated_y,
        preallocated_d,
        observation_noise,
        length(y),
        capacity,
        containers
    )
end

function Surrogate(
    ψ::RadialBasisFunction,
    dim::Int,
    capacity::Int=DEFAULT_CAPACITY,
    observation_noise::T=1e-6) where T <: Real
    preallocated_X = zeros(T, dim, capacity)
    preallocated_K = zeros(T, capacity, capacity)
    preallocated_Kscratch = zeros(T, capacity, capacity)
    preallocated_L = ElasticPDMat(capacity=capacity)
    preallocated_d = zeros(T, capacity)
    preallocated_y = zeros(T, capacity)
    UNUSED_NUMBER_OF_BASIS_FUNCTIONS = 1

    containers = PreallocatedContainers{T}(
        UNUSED_NUMBER_OF_BASIS_FUNCTIONS,
        dim,
        capacity,
        length(ψ),
        0
    )

    return Surrogate(
        ψ,
        preallocated_X,
        preallocated_K,
        preallocated_Kscratch,
        preallocated_L,
        preallocated_y,
        preallocated_d,
        observation_noise,
        0,
        capacity,
        containers
    )
end

get_active_KxX(s::AbstractSurrogate) = @view s.containers.KxX[1:get_observed(s)]
get_active_grad_KxX(s::AbstractSurrogate) = @view s.containers.grad_KxX[:, 1:get_observed(s)]
get_active_δKXX(s::AbstractSurrogate) = @view s.containers.δKXX[1:get_observed(s), 1:get_observed(s)]
get_Hk(s::AbstractSurrogate) = @view s.containers.Hk[:, :]
get_Hσ(s::AbstractSurrogate) = @view s.containers.Hσ[:, :]
get_Hf(s::AbstractSurrogate) = @view s.containers.Hf[:, :]
get_diff_x(s::AbstractSurrogate) = @view s.containers.diff_x[:]
get_Hz(s::AbstractSurrogate) = @view s.containers.Hz[:, :]
get_px(s::HybridSurrogate) = @view s.containers.px[:, :]
get_grad_px(s::HybridSurrogate) = @view s.containers.grad_px[:, :]
get_zz(s::HybridSurrogate) = @view s.containers.zz[:, :]


function set_kernel!(s::Surrogate, kernel::RadialBasisFunction)
    # println("Factors Before: ", view(s.L.chol).factors)
    @views begin
        N = get_observed(s)
        s.ψ = kernel
        observation_noise = get_observation_noise(s)
        # Grab the buffer and update inplace
        @timeit to "Assign Buffer" Kbuffer = s.L.mat
        @timeit to "Set Kernel KXX" eval_KXX!(
            kernel,
            get_active_covariates(s),
            Kbuffer,
            get_diff_x(s)
        )
        # Ensures PSDness
        @timeit to "Jitter Diag" for jj in 1:N Kbuffer[jj, jj] += (JITTER + observation_noise) end
        copyto!(
            s.K[1:N, 1:N],
            Kbuffer
        )
        @timeit to "Scratchpad Assignment" copyto!(
            view(s.L.chol).factors,
            Kbuffer
        )
        @timeit to "Compute Cholesky" cholesky!(Symmetric(view(s.L.chol).factors))
        # cholesky!(view(s.L.chol).factors)
        # s.d[1:N] = s.L \ get_active_observations(s)
        # s.L \ get_active_observations(s)
        @timeit to "Copy Observations" yc = copy(get_active_observations(s))
        @timeit to "Inplace Coefficient Solve" ldiv!(s.L, yc)
        @timeit to "Assign Coefficient Solve" s.d[1:N] .= yc
    end
    # println("Factors After: ", view(s.L.chol).factors)
end


function set_kernel!(s::HybridSurrogate, kernel::RadialBasisFunction)
    @views begin
        N = get_observed(s)
        s.ψ = kernel
        observation_noise = get_observation_noise(s)
        # s.K[1:N, 1:N] = eval_KXX(kernel, get_active_covariates(s)) + (JITTER + σn2) * I
        # Grab the buffer and update inplace
        Kbuffer = s.L.mat
        eval_KXX!(
            kernel,
            get_active_covariates(s),
            Kbuffer,
            get_diff_x(s)
        )
        # Ensure PSDness
        for jj in 1:N Kbuffer[jj, jj] += (JITTER + observation_noise) end
        copyto!(
            s.K[1:N, 1:N],
            Kbuffer
        )
        copyto!(
            view(s.L.chol).factors,
            Kbuffer
        )
        cholesky!(Symmetric(view(s.L.chol).factors))
        # s.L[1:N, 1:N] = LowerTriangular(
        #     cholesky(
        #         Hermitian(get_active_covariance(s))
        #     ).L
        # )
        @timeit to "Hybrid Coefficient Solve" d, λ = coefficient_solve(
            view(s.L.chol).L,
            get_active_parametric_basis_matrix(s),
            get_active_observations(s),
            get_schur_buffer(s)
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
        s.containers.schur.active_index = N
        observation_noise = get_observation_noise(s)
        eval_KXX!(
            get_kernel(s),
            get_active_covariates(s),
            get_active_covariance(s),
            get_diff_x(s)
        )
        # Ensures PDSness
        for jj in 1:N
            get_active_covariance(s)[jj, jj] += (JITTER + observation_noise)
        end
        copyto!(s.Kscratch[1:N, 1:N], get_active_covariance(s))
        # Grab the active covariance and write a copy of the covariance matrix
        # tin our buffer o it
        s.L = ElasticPDMat(
            convert(Matrix, s.Kscratch[1:N, 1:N]),
            capacity=s.capacity
        )
        yc = copy(y)
        ldiv!(s.L, yc)
        s.d[1:N] = yc
        s.y[1:N] = y
    end
end


function set!(s::HybridSurrogate, X::Matrix{T}, y::Vector{T}) where {T<:Real}
    @views begin
        dim, N = size(X)

        s.X[:, 1:N] = X
        s.observed = N
        s.containers.schur.active_index = N
        observation_noise = get_observation_noise(s)
        eval_KXX!(
            get_kernel(s),
            get_active_covariates(s),
            get_active_covariance(s),
            get_diff_x(s)
        )
        # Ensures PDSness
        for jj in 1:N
            get_active_covariance(s)[jj, jj] += (JITTER + observation_noise)
        end
        # s.K[1:N, 1:N] += (JITTER + observation_noise) * I
        copyto!(s.Kscratch[1:N, 1:N], get_active_covariance(s))
        # s.L[1:N, 1:N] = LowerTriangular(
        #     cholesky(
        #         Hermitian(
        #             get_active_covariance(s)
        #         )
        #     ).L
        # )
        s.L = ElasticPDMat(
            convert(Matrix, s.Kscratch[1:N, 1:N]),
            capacity=capacity
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
            view(s.L.chol).L,
            get_active_parametric_basis_matrix(s),
            y,
            get_schur_buffer(s)
        )
        s.d[1:N] = d
        s.λ[1:length(get_parametric_basis_function(s))] = λ
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
        get_observation_noise(s)
    )
end


function resize(s::Surrogate)
    return Surrogate(
        get_kernel(s),
        get_covariates(s),
        get_observations(s),
        get_capacity(s) * DOUBLE,
        get_observation_noise(s)
    )
end


function insert!(s::AbstractSurrogate, x::Vector{T}, y::T) where {T<:Real}
    insert_index = get_observed(s) + 1
    s.X[:, insert_index] = x
    s.y[insert_index] = y
end

function insert!(s::HybridSurrogate, x::Vector{T}, y::T) where { T <: Real }
    insert_index = get_observed(s) + 1
    s.containers.schur.active_index += 1
    s.X[:, insert_index] = x
    s.y[insert_index] = y 
end


function update_covariance!(s::HybridSurrogate, x::Vector{T}) where {T<:Real}
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
    append!(s.L, vec(s.K[1:update_index, update_index]))
end

function update_covariance!(s::Surrogate, x::Vector{T}) where {T<:Real}
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
    append!(s.L, vec(s.K[1:update_index, update_index]))
end

function update_cholesky!(s::AbstractSurrogate)
    # Grab entries from update covariance matrix
    @views begin
        n = get_observed(s)
        B = s.K[n:n, 1:n-1]
        C = s.K[n:n, n:n] .+ (JITTER + get_observation_noise(s))
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
    d, λ = coefficient_solve(
        view(s.L.chol).L,
        PX,
        y,
        get_schur_buffer(s)
    )
    @views begin
        s.d[1:length(d)] = d
        s.λ[:] = λ
    end
end


function update_coefficients!(s::Surrogate)
    update_index = get_observed(s)
    @views begin
        # L = s.L[1:update_index, 1:update_index]
        # s.d[1:update_index] = L' \ (L \ s.y[1:update_index])
        s.d[1:update_index] = s.L \ s.y[1:update_index]
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
    # update_cholesky!(s) # Updates cholesky factorization matrix L
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
    # update_cholesky!(s)
    update_coefficients!(s)
    return s
end

function predictive_mean(s::HybridSurrogate, x::AbstractVector{<:Real})
    d = get_active_coefficients(s)
    kx = eval_KxX!(
        get_kernel(s),
        x,
        get_active_covariates(s),
        get_active_KxX(s),
        get_diff_x(s)
    )
    λ = get_parametric_component_coefficients(s)
    px = eval_basis!(
        get_parametric_basis_function(s),
        x,
        get_px(s)
    )
    return dot(kx, d) + dot(px, λ)
end

function predictive_mean_gradient(s::HybridSurrogate, x::AbstractVector{<:Real})
    d = get_active_coefficients(s)
    λ = get_parametric_component_coefficients(s)
    ∇kx = eval_∇KxX!(
        get_kernel(s),
        x,
        get_active_covariates(s),
        get_active_grad_KxX(s),
        get_diff_x(s)
    )
    ∇px = eval_∇basis!(
        get_parametric_basis_function(s),
        x,
        get_grad_px(s)
    )
    return ∇kx * d + ∇px * λ
end

@timeit to "Hybrid Predictive Std" function predictive_std(s::HybridSurrogate, x::AbstractVector{<:Real})
    kx = eval_KxX!(
        get_kernel(s),
        x,
        get_active_covariates(s),
        get_active_KxX(s),
        get_diff_x(s)
    )
    px = eval_basis!(
        get_parametric_basis_function(s),
        x,
        get_px(s)
    )
    v = vcat(vec(px), kx)
    P = get_active_parametric_basis_matrix(s)   
    w = schur_reduced_system_solve(
        get_active_cholesky(s),
        P,
        px,
        kx,
        get_schur_buffer(s)
    )
     
    k0 = get_kernel(s)(0.)
    return sqrt(k0 - dot(v, w))
end

@timeit to "Hybrid Predictive Std Gradient" function predictive_std_gradient(s::HybridSurrogate, x::AbstractVector{<:Real})
    P = get_active_parametric_basis_matrix(s)
    K = get_active_covariance(s)
    kx = eval_KxX!(
        get_kernel(s),
        x,
        get_active_covariates(s),
        get_active_KxX(s),
        get_diff_x(s)
    )
    px = eval_basis!(
        get_parametric_basis_function(s),
        x,
        get_px(s)
    )
    ∇px = eval_∇basis!(
        get_parametric_basis_function(s),
        x,
        get_grad_px(s)
    )
    ∇kx = eval_∇KxX!(
        get_kernel(s),
        x,
        get_active_covariates(s),
        get_active_grad_KxX(s),
        get_diff_x(s)
    )
    w = schur_reduced_system_solve(
        get_active_cholesky(s),
        P,
        px,
        kx,
        get_schur_buffer(s)
    )
    ∇v = hcat(∇px, ∇kx)
    σ = predictive_std(s, x)
    return -(∇v * w) / σ
end

function predictive_mean(s::Surrogate, x::AbstractVector{<:Real})
    c = get_active_coefficients(s)
    kx = eval_KxX!(
        get_kernel(s),
        x,
        get_active_covariates(s),
        get_active_KxX(s),
        get_diff_x(s)
    )
    return dot(kx, c)
end

function predictive_mean_gradient(s::Surrogate, x::AbstractVector{<:Real})
    c = get_active_coefficients(s)
    ∇kx = eval_∇KxX!(
        get_kernel(s),
        x,
        get_active_covariates(s),
        get_active_grad_KxX(s),
        get_diff_x(s)
    )
    return ∇kx * c
end

function predictive_std(s::Surrogate, x::AbstractVector{<:Real})
    k0 = get_kernel(s)(0.)
    kx = eval_KxX!(
        get_kernel(s),
        x,
        get_active_covariates(s),
        get_active_KxX(s),
        get_diff_x(s)
    )
    w = copy(kx)
    ldiv!(s.L, w)
    return sqrt(k0 - dot(kx, w))
end

function predictive_std_gradient(s::Surrogate, x::AbstractVector{<:Real})
    kx = eval_KxX!(
        get_kernel(s),
        x,
        get_active_covariates(s),
        get_active_KxX(s),
        get_diff_x(s)
    )
    ∇kx = eval_∇KxX!(
        get_kernel(s),
        x,
        get_active_covariates(s),
        get_active_grad_KxX(s),
        get_diff_x(s)
    )
    σ = predictive_std(s, x)
    w = copy(kx)
    ldiv!(s.L, w)
    return -(∇kx * w) / σ
end

compute_moments(s::AbstractSurrogate, x) = (predictive_mean(s, x), predictive_std(s, x))
compute_moments_gradient(s::AbstractSurrogate, x) = (predictive_mean_gradient(s, x), predictive_std_gradient(s, x))


# ------------------------------------------------------------------
# Operations for computing optimal hyperparameters.
# ------------------------------------------------------------------
function logabsdet_M(s::HybridSurrogate)
    # Get active pieces
    L = get_active_cholesky(s)         # LowerTriangular from cholesky(K)
    K = get_active_covariance(s)       # n × n
    P = get_active_parametric_basis_matrix(s)  # n × m
    zz = get_zz(s)                     # m × m

    # Step 1: log|det(K)|
    diagL = diag(L)
    logabsdetK = 2 * sum(log, diagL)

    # Step 2: form S = zz - P^T K^{-1} P
    Y = get_Y(get_schur_buffer(s))
    copyto!(Y, P)
    ldiv!(LowerTriangular(L), Y)
    
    S = get_S(get_schur_buffer(s))
    copyto!(S, zz)
    mul!(S, transpose(Y), Y, -1., 1.)

    # Step 3: log|det(S)|
    logabsdetS = log(abs(det(S)))

    return logabsdetK + logabsdetS
end

function log_likelihood(s::HybridSurrogate)
    n = get_observed(s)
    m = length(get_parametric_basis_function(s))
    yz = [get_active_observations(s); zeros(m)]
    d = get_active_coefficients(s)
    λ = get_parametric_component_coefficients(s)
    dλ = vcat(d, λ)
    P = get_active_parametric_basis_matrix(s)
    K = get_active_covariance(s)
    zz = get_zz(s)

    ladM = logabsdet_M(s)
    # M = Matrix{Float64}(
    #     [zz P';
    #      P  K])
    # ladM = log(abs(det(M)))

    return -dot(yz, dλ) / 2 - n * log(2π) / 2 - ladM
end


function log_likelihood(s::Surrogate)
    n = get_observed(s)
    y = get_active_observations(s)
    c = get_active_coefficients(s)
    L = s.L
    return -(y' * c) / 2 - logdet(L) / 2 - (n * log(2π) / 2)
end


function optimize!(
    s::AbstractSurrogate;
    lowerbounds::Vector{T},
    upperbounds::Vector{T},
    restarts::Int = 32) where T
    n = length(lowerbounds)
    # opt = NLopt.Opt(:LD_SLSQP, n)
    opt = NLopt.Opt(:LD_LBFGS, n)
    
    # Set the lower and upper bounds
    lower_bounds!(opt, lowerbounds)
    upper_bounds!(opt, upperbounds)
    
    # Set stopping criteria
    xtol_rel!(opt, 1e-4)
    ftol_rel!(opt, 1e-4)
    maxeval!(opt, 100)             # maximum number of evaluations/iterations
    # Optionally, set a time limit if NEWTON_SOLVE_TIME_LIMIT is defined:
    maxtime!(opt, NEWTON_SOLVE_TIME_LIMIT)

    function nlopt_obj(θ, grad)
        # if length(grad) > 0
        #     @timeit to "Grad Log Likelihood" grad[:] = ∇log_likelihood(s)
        # end
        @timeit to "Set Kernel Hypers" set_kernel!(s, set_hyperparameters!(get_kernel(s), θ))
        @timeit to "Log Likelihood" res = log_likelihood(s)
        return res
    end
    
    maxf = -Inf
    maxθ = lowerbounds
    seq = ScaledLHSIterator(lowerbounds, upperbounds, restarts)

    for θ in seq
        NLopt.max_objective!(opt, nlopt_obj)
        @timeit to "NLopt Hypers" f_max, θ_max, ret = NLopt.optimize(
            opt, convert(Vector, θ)
        )
        if f_max > maxf
            maxf = f_max
            maxθ = θ_max
        end
    end
    set_kernel!(s, set_hyperparameters!(get_kernel(s), maxθ))
    return nothing
end