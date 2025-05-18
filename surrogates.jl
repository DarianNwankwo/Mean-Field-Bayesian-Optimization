mutable struct SurrogateEvaluationCache # <: AbstractEvaluationCache
    x::Vector{Float64}          # Last input used for evaluation
    μ::Float64                  # Cached mean
    σ::Float64                  # Cached standard deviation
    ∇μ::Vector{Float64}         # Cached gradient of the mean
    ∇σ::Vector{Float64}         # Cached gradient of the standard deviation
    valid::Bool                 # Flag indicating if the cache is valid
    updates::Int64
end

# Constructor for the cache, assuming dimension `d` for the input x.
function SurrogateEvaluationCache(d::Int)
    return SurrogateEvaluationCache(
        zeros(d),
        0.0,
        0.0,
        zeros(d),
        zeros(d),
        false,
        0
    )
end
invalidate!(cache::SurrogateEvaluationCache) = cache.valid = false
function should_reuse_computation(
    cache::SurrogateEvaluationCache,
    x::AbstractVector{<:Real},
    atol=CACHE_SAME_X_TOLERANCE)
    return cache.valid && all(abs.(x .- cache.x) .< atol)
end  

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
    schur::SchurBuffer,
    x::AbstractVector{<:Real})
    if schur.valid && isequal(x, schur.x)
        return schur.w
    end
    # Preallocate scratch buffers
    n, m = size(PX)
    Y = get_Y(schur)
    S = get_S(schur)
    tmp = get_tmp(schur)
    r = get_r(schur)

    # term1: Y = L \ PX
    copyto!(Y, PX)    
    ldiv!(LowerTriangular(L), Y)


    # term2: S = Y' * Y
    mul!(S, transpose(Y), Y)

    # term3: r = Y'*(L \ KxX) - Px
    copyto!(tmp, KxX)
    ldiv!(LowerTriangular(L), tmp)
    mul!(r, transpose(Y), tmp)
    r .-= Px

    # term4: Cholesky factor of S
    fS = cholesky!(Hermitian(S))

    # term5: w1 = fS \ r  (in-place solve)
    w1 = get_w1(schur)
    copyto!(w1, r)
    ldiv!(fS, w1)

    # term6: w0 = L'\(L\(KxX - PX*w1))
    w0 = get_w0(schur)
    copyto!(w0, KxX)
    mul!(w0, PX, w1, -1.0, 1.0)     # w0 = KxX - PX*w1
    ldiv!(LowerTriangular(L), w0)
    ldiv!(UpperTriangular(L'), w0)

    # term7: return concatenated solution
    w = get_w(schur)
    w[1:length(w1)] .= w1
    w[length(w1)+1:length(w1) + schur.active_index] .= w0
    schur.valid = true
    return w
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
    PM <: ElasticPDMat,
    T <: Real} <: AbstractSurrogate
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
    containers::PreallocatedContainers{T}
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


function coefficient_solve(
    L::ElasticPDMat,
    PX::AbstractMatrix{<:Real},
    fx::AbstractVector{<:Real},
    schur::SchurBuffer)
    d = get_d(schur)
    λ = get_λ(schur)
    Y = get_Y(schur)
    S = get_S(schur)
    r = get_r(schur)
    tmp = get_tmp(schur)

    chol_view = view(L.chol)
    copyto!(Y, PX)
    ldiv!(chol_view, Y) 
    # S = Y' * Y
    # mul!(S, transpose(Y), Y)
    mul!(S, transpose(PX), Y)

    # r = PX' * (L' \ (L \ fx)), all in-place
    copyto!(tmp, fx)
    ldiv!(chol_view, tmp)

    # Solve S * λ = r in-place
    mul!(r, transpose(PX), tmp)
    fS = cholesky!(Hermitian(S))
    copyto!(λ, r)
    ldiv!(fS, λ)

    # 5) Solve for d = L' \ (L \ (fx - PX*λ)) in-place
    copyto!(tmp, fx)
    mul!(tmp, PX, λ, -1.0, 1.0)
    ldiv!(chol_view, tmp)
    copyto!(d, tmp)

    return nothing
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
    coefficient_solve(
        preallocated_L,
        PX,
        y,
        containers.schur
    )

    preallocated_d = zeros(T, capacity)
    preallocated_d[1:length(get_d(containers.schur))] = get_d(containers.schur)

    λ_polynomial = zeros(T, length(get_λ(containers.schur)))
    λ_polynomial[:] = get_λ(containers.schur)

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
    containers = PreallocatedContainers{T}(
        length(ϕ), dim, capacity, length(ψ), observed
    )

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

mutable struct Surrogate{
    RBF <: StationaryKernel,
    PM <: ElasticPDMat,
    T <: Real} <: AbstractSurrogate
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
    containers::PreallocatedContainers{T}
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
get_diff_x(s::AbstractSurrogate) = s.containers.diff_x
get_Hz(s::AbstractSurrogate) = @view s.containers.Hz[:, :]
get_yc(s::AbstractSurrogate) = @view s.containers.yc[1:get_observed(s)]
get_px(s::HybridSurrogate) = @view s.containers.px[:, :]
get_grad_px(s::HybridSurrogate) = @view s.containers.grad_px[:, :]
get_zz(s::HybridSurrogate) = @view s.containers.zz[:, :]
get_w(s::AbstractSurrogate) = @view s.containers.w[1:get_observed(s)]


function set_kernel!(s::Surrogate, kernel::RadialBasisFunction)
    # println("Factors Before: ", view(s.L.chol).factors)
    @views begin
        N = get_observed(s)
        s.ψ = kernel
        observation_noise = get_observation_noise(s)
        # Grab the buffer and update inplace
        K = get_active_covariance(s)
        Kbuffer = s.L.mat
        eval_KXX!(
            kernel,
            get_active_covariates(s),
            K,
            get_diff_x(s)
        )
        # Ensures PSDness
        for jj in 1:N K[jj, jj] += (JITTER + observation_noise) end
        copyto!(
            # s.K[1:N, 1:N],
            # Kbuffer
            Kbuffer,
            K
        )
        copyto!(
            view(s.L.chol).factors,
            Kbuffer
        )
        cholesky!(Symmetric(view(s.L.chol).factors))
        yc = get_yc(s)
        copyto!(yc, get_active_observations(s))
        ldiv!(s.L, yc)
        s.d[1:N] .= yc
    end
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
        schur_buffer = get_schur_buffer(s)
        coefficient_solve(
            s.L,
            get_active_parametric_basis_matrix(s),
            get_active_observations(s),
            schur_buffer
        )
        s.d[1:N] = get_d(schur_buffer)
        s.λ[:] = get_λ(schur_buffer)
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
        ldiv!(s.L.chol, yc)
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
            capacity=s.capacity
        )
        eval_basis!(
            get_parametric_basis_function(s),
            get_active_covariates(s),
            get_active_parametric_basis_matrix(s)
        )
            # s.P[1:N, 1:length(ϕ)])
        # PX = s.P[1:N, 1:length(ϕ)]
        # s.P[1:N, 1:length(ϕ)] = PX
        schur_buffer = get_schur_buffer(s)
        coefficient_solve(
            # view(s.L.chol).L,
            s.L,
            get_active_parametric_basis_matrix(s),
            y,
            get_schur_buffer(s)
        )
        s.d[1:N] = get_d(schur_buffer)
        s.λ[1:length(get_parametric_basis_function(s))] = get_λ(schur_buffer)
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
            get_diff_x(s)
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
            get_diff_x(s)
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
    schur_buffer = get_schur_buffer(s)
    coefficient_solve(
        s.L,
        PX,
        y,
        schur_buffer
    )
    @views begin
        s.d[1:length(get_d(schur_buffer))] = get_d(schur_buffer)
        s.λ[:] = get_λ(schur_buffer)
    end
end


function update_coefficients!(s::Surrogate)
    update_index = get_observed(s)
    @views begin
        yc = get_yc(s)
        copyto!(yc, s.y[1:update_index])
        ldiv!(s.L, yc)
        # s.d[1:update_index] = s.L \ s.y[1:update_index]
        s.d[1:update_index] .= yc
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
    update_coefficients!(s)
    return s
end

function predictive_mean(
    s::HybridSurrogate,
    x::AbstractVector{<:Real},
    cache::SurrogateEvaluationCache;
    atol=CACHE_SAME_X_TOLERANCE)
    if should_reuse_computation(cache, x, atol) return cache.μ end
    d = get_active_coefficients(s)
    kx = eval_KxX!(get_kernel(s), x, get_active_covariates(s), get_active_KxX(s), get_diff_x(s))
    λ = get_parametric_component_coefficients(s)
    px = eval_basis!(get_parametric_basis_function(s), x, get_px(s))
    cache.μ = dot(kx, d) + dot(px, λ)
    return cache.μ
end

function predictive_mean_gradient!(
    s::HybridSurrogate,
    x::AbstractVector{<:Real},
    cache::SurrogateEvaluationCache;
    atol=CACHE_SAME_X_TOLERANCE)
    if should_reuse_computation(cache, x, atol) return cache.∇μ end
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
    # Clear out
    fill!(cache.∇μ, 0.0)
    # out = ∇kx * d
    mul!(cache.∇μ, ∇kx, d)
    # out += ∇px * λ
    mul!(cache.∇μ, ∇px, λ, 1.0, 1.0)
    return cache.∇μ
end

function predictive_std(
    s::HybridSurrogate,
    x::AbstractVector{<:Real},
    cache::SurrogateEvaluationCache;
    atol=CACHE_SAME_X_TOLERANCE)
    if should_reuse_computation(cache, x, atol) return cache.σ end
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
    P = get_active_parametric_basis_matrix(s)   
    w = schur_reduced_system_solve(
        get_active_cholesky(s),
        P,
        px,
        kx,
        get_schur_buffer(s),
        x
    )
    w0 = get_w0(get_schur_buffer(s))
    w1 = get_w1(get_schur_buffer(s))
    # TODO: Can do inplace operations here with mul!
    dot_px = dot(px, w1)
    dot_kx = dot(kx, w0)
    k0 = get_kernel(s)(0.)
    # return sqrt(k0 - dot(v, w))
    cache.σ = sqrt(k0 - (dot_px + dot_kx))
    return cache.σ
end

function predictive_std_gradient!(
    s::HybridSurrogate,
    x::AbstractVector{<:Real},
    cache::SurrogateEvaluationCache;
    atol=CACHE_SAME_X_TOLERANCE)
    if should_reuse_computation(cache, x, atol) return cache.∇σ end
    @views begin
    P = get_active_parametric_basis_matrix(s)
    K = get_active_covariance(s)
    kx = eval_KxX!(get_kernel(s), x, get_active_covariates(s), get_active_KxX(s), get_diff_x(s))
    px = eval_basis!(get_parametric_basis_function(s), x, get_px(s))
    ∇px = eval_∇basis!(get_parametric_basis_function(s), x, get_grad_px(s))
    ∇kx = eval_∇KxX!(get_kernel(s), x, get_active_covariates(s), get_active_grad_KxX(s), get_diff_x(s))
    w = schur_reduced_system_solve(get_active_cholesky(s), P, px, kx, get_schur_buffer(s), x)
    dim = length(cache.∇σ)
    w0 = get_w0(get_schur_buffer(s))
    w1 = get_w1(get_schur_buffer(s))
    # ∇σ = ∇px * w1
    for i in 1:size(∇px, 1) cache.∇σ[i] = dot(∇px[i, :], w1) end
    # ∇σ = ∇px * w1 + ∇kx * w0
    for i in 1:size(∇kx, 1) cache.∇σ[i] += dot(∇kx[i, :], w0) end
    σ = predictive_std(s, x, cache)
    
    # ∇σ = (∇px * w1 + ∇kx * w0) / σ
    @inbounds @simd for i in 1:dim cache.∇σ[i] = cache.∇σ[i] / σ end
    # ∇σ = -(∇px * w1 + ∇kx * w0) / σ
    cache.∇σ .*= -1.
    return cache.∇σ
    end
end

function predictive_mean(
    s::Surrogate,
    x::AbstractVector{<:Real},
    cache::SurrogateEvaluationCache;
    atol=CACHE_SAME_X_TOLERANCE)
    if should_reuse_computation(cache, x, atol) return cache.μ end
    c = get_active_coefficients(s)
    kx = eval_KxX!(get_kernel(s), x, get_active_covariates(s), get_active_KxX(s), get_diff_x(s))
    cache.μ = dot(kx, c)
    return dot(kx, c)
end

function predictive_mean_gradient!(
    s::Surrogate,
    x::AbstractVector{<:Real},
    cache::SurrogateEvaluationCache;
    atol=CACHE_SAME_X_TOLERANCE)
    if should_reuse_computation(cache, x, atol) return cache.∇μ end
    c = get_active_coefficients(s)
    ∇kx = eval_∇KxX!(get_kernel(s), x, get_active_covariates(s), get_active_grad_KxX(s), get_diff_x(s))
    mul!(cache.∇μ, ∇kx, c)
    return cache.∇μ
end

function predictive_std(
    s::Surrogate,
    x::AbstractVector{<:Real},
    cache::SurrogateEvaluationCache;
    atol=CACHE_SAME_X_TOLERANCE)
    if should_reuse_computation(cache, x, atol) return cache.σ end
    k0 = get_kernel(s)(0.)
    kx = eval_KxX!(get_kernel(s), x, get_active_covariates(s), get_active_KxX(s), get_diff_x(s))
    w = get_w(s)
    copyto!(w, kx)
    ldiv!(s.L, w)
    cache.σ = sqrt(k0 - dot(kx, w))
    return cache.σ
end

function predictive_std_gradient!(
    s::Surrogate,
    x::AbstractVector{<:Real},
    cache::SurrogateEvaluationCache;
    atol=CACHE_SAME_X_TOLERANCE)
    if should_reuse_computation(cache, x, atol) return cache.∇σ end
    kx = eval_KxX!(get_kernel(s), x, get_active_covariates(s), get_active_KxX(s), get_diff_x(s))
    ∇kx = eval_∇KxX!(get_kernel(s), x, get_active_covariates(s), get_active_grad_KxX(s), get_diff_x(s))
    σ = predictive_std(s, x, cache, atol=atol)
    w = get_w(s)
    copyto!(w, kx)
    ldiv!(s.L, w)
    mul!(cache.∇σ, ∇kx, w)
    cache.∇σ ./= σ
    cache.∇σ .*= -1.
    return cache.∇σ
end

function compute_moments!(
    s::AbstractSurrogate,
    x::AbstractVector{<:Real},
    cache::SurrogateEvaluationCache;
    atol=CACHE_SAME_X_TOLERANCE)
    μ = predictive_mean(s, x, cache, atol=atol)
    σ = predictive_std(s, x, cache, atol=atol)
    return (μ, σ)
end


function compute_moments_gradient!(
    s::AbstractSurrogate,
    x::AbstractVector{<:Real},
    cache::SurrogateEvaluationCache;
    atol=CACHE_SAME_X_TOLERANCE)
    ∇μ = predictive_mean_gradient!(s, x, cache, atol=atol)
    ∇σ = predictive_std_gradient!(s, x, cache, atol=atol)

    return (∇μ, ∇σ)
end


# ------------------------------------------------------------------
# Operations for computing optimal hyperparameters.
# ------------------------------------------------------------------
function logabsdet_M2(s::HybridSurrogate)
    # Get active pieces
    L = get_active_cholesky(s)         # LowerTriangular from cholesky(K)
    K = get_active_covariance(s)       # n × n
    P = get_active_parametric_basis_matrix(s)  # n × m
    zz = get_zz(s)                     # m × m

    # Step 1: log|det(K)|
    # diagL = diag(L)
    # logabsdetK = 2 * sum(log, diagL)
    logabsdetK = 0.
    for i in 1:size(L, 2)
        logabsdetK += log(L[i, i])
    end
    logabsdetK *= 2.

    # Step 2: form S = zz - P^T K^{-1} P = zz - (P^T L^-T) (L ^-1 P)
    Y = get_Y(get_schur_buffer(s))
    copyto!(Y, P)
    ldiv!(LowerTriangular(L), Y)
    
    S = get_S(get_schur_buffer(s))
    # copyto!(S, zz)
    # mul!(S, transpose(Y), Y, -1., 1.)
    mul!(S, transpose(Y), Y)
    S .*= -1.

    # Step 3: log|det(S)|
    logabsdetS = log(abs(det(S)))

    return logabsdetK + logabsdetS
end


function log_likelihood(s::HybridSurrogate)
    n = get_observed(s)
    m = length(get_parametric_basis_function(s))
    # yz = [get_active_observations(s); zeros(m)]
    yz = get_active_observations(s)
    d = get_active_coefficients(s)
    λ = get_parametric_component_coefficients(s)
    # dλ = vcat(d, λ)
    P = get_active_parametric_basis_matrix(s)
    K = get_active_covariance(s)
    zz = get_zz(s)

    ladM = logabsdet_M(s)
    # M = Matrix{Float64}(
    #     [zz P';
    #      P  K])
    # ladM = log(abs(det(M)))

    # return -dot(yz, dλ) / 2 - n * log(2π) / 2 - ladM
    return -dot(yz, d) / 2 - n * log(2π) / 2 - ladM
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
    xtol_rel!(opt, X_RELTOL)
    ftol_rel!(opt, F_RELTOL)
    maxeval!(opt, 100)             # maximum number of evaluations/iterations
    # Optionally, set a time limit if NEWTON_SOLVE_TIME_LIMIT is defined:
    maxtime!(opt, NEWTON_SOLVE_TIME_LIMIT)

    function nlopt_obj(θ, grad)
        # if length(grad) > 0
        #     @timeit to "Grad Log Likelihood" grad[:] = ∇log_likelihood(s)
        # end
        set_hyperparameters!(get_kernel(s), θ)
        set_kernel!(s, get_kernel(s))
        res = log_likelihood(s)
        return res
    end
    
    maxf = -Inf
    maxθ = lowerbounds
    seq = ScaledLHSIterator(lowerbounds, upperbounds, restarts)

    for θ in seq
        NLopt.max_objective!(opt, nlopt_obj)
        f_max, θ_max, ret = NLopt.optimize(
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