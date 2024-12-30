get_kernel(afs::AbstractSurrogate) = afs.ψ
get_covariates(afs::AbstractSurrogate) = afs.X
get_observations(afs::AbstractSurrogate) = afs.y
get_decision_rule(afs::AbstractSurrogate) = afs.g
get_coefficients(afs::AbstractSurrogate) = afs.d
get_parametric_component_coefficients(afs::AbstractSurrogate) = afs.λ
get_cholesky(as::AbstractSurrogate) = as.L
get_covariance(as::AbstractSurrogate) = as.K
get_parametric_basis_matrix(as::AbstractSurrogate) = as.P


"""
The model we consider consists of some parametric and nonparametric component, i.e.
f(x) ∼ P(x)c + Z(x) where Z ∼ GP(0, k_h), covariance function k: Ω × Ω → R, multivariate 
normal c ∼ N(0, Σ) with Σ = (1/ϵ)Σ_{ref}^2, and P: Ω → R^m.

We need to maintain information about the covariance function, which is our kernel,
and the mechanism that allows us to construct our linear system given some polynomial
basis.
"""
mutable struct HybridSurrogate{RBF <: StationaryKernel, PBF <: ParametricRepresentation} <: AbstractSurrogate
    ψ::RBF
    ϕ::PBF
    X::Matrix{Float64}
    P::Matrix{Float64}
    K::Matrix{Float64}
    L::LowerTriangular{Float64, Matrix{Float64}}
    y::Vector{Float64}
    d::Vector{Float64}
    λ::Vector{Float64}
    observation_noise::Float64
    g::DecisionRule
    observed::Int
    capacity::Int
end

mutable struct GaussianProcess{RBF <: StationaryKernel} <: AbstractSurrogate
    ψ::RBF
    X::Matrix{Float64}
    K::Matrix{Float64}
    L::LowerTriangular{Float64, Matrix{Float64}}
    y::Vector{Float64}
    d::Vector{Float64}
    observation_noise::Float64
    g::DecisionRule
    observed::Int
    capacity::Int
end

get_capacity(s::AbstractSurrogate) = s.capacity
increment!(s::AbstractSurrogate) = s.observed += 1
get_observed(s::AbstractSurrogate) = s.observed
is_full(s::AbstractSurrogate) = get_observed(s) == get_capacity(s)
get_active_covariates(s::AbstractSurrogate) = @view get_covariates(s)[:, 1:get_observed(s)]
get_active_cholesky(s::AbstractSurrogate) = @view get_cholesky(s)[1:get_observed(s), 1:get_observed(s)]
get_active_covariance(s::AbstractSurrogate) = @view get_covariance(s)[1:get_observed(s), 1:get_observed(s)]
get_active_observations(s::AbstractSurrogate) = @view get_observations(s)[1:get_observed(s)]
get_active_coefficients(s::AbstractSurrogate) = @view get_coefficients(s)[1:get_observed(s)]
get_active_parametric_basis_matrix(s::AbstractSurrogate) = @view get_parametric_basis_matrix(s)[1:get_observed(s), :]
get_parametric_basis_function(s::AbstractSurrogate) = s.ϕ
get_observation_noise(s::AbstractSurrogate) = s.observation_noise


# Define the custom show method for Surrogate
function Base.show(io::IO, s::HybridSurrogate{RBF}) where {RBF}
    print(io, "HybridSurrogate{RBF = ")
    show(io, s.ψ)    # Use the show method for RBF
    print(io, "}")
end

get_decision_rule(s::AbstractSurrogate) = s.g
set_decision_rule!(s::AbstractSurrogate, g::DecisionRule) = s.g = g

function coefficient_solve(
    KXX::AbstractMatrix{T1},
    PX::AbstractMatrix{T2},
    observation_noise::Float64,
    y::AbstractVector{T3}) where {T1 <: Real, T2 <: Real, T3 <: Real}
    p_dim, k_dim = size(PX, 2), size(KXX, 2)
    A = [KXX PX;
         PX' zeros(p_dim, p_dim)]
    A = A + observation_noise * I
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
    capacity::Int = DEFAULT_CAPACITY,
    decision_rule::AbstractDecisionRule = EI(),
    observation_noise::T = 1e-6) where T <: Real
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
    KXX = eval_KXX(ψ, X)
    preallocated_K[1:N, 1:N] = KXX + observation_noise * I

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
    d, λ = coefficient_solve(KXX, PX, observation_noise, y)

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

function GaussianProcess(
    ψ::RadialBasisFunction,
    X::Matrix{T},
    y::Vector{T};
    capacity::Int = DEFAULT_CAPACITY,
    decision_rule::AbstractDecisionRule = EI(),
    observation_noise::T = 1e-6) where T <: Real
    @assert length(y) <= capacity "Capacity must be >= number of observations."
    d, N = size(X)

    """
    Preallocate a matrix for covariates of size d x capacity where capacity is the maximum
    number of observations.
    """
    preallocated_X = zeros(d, capacity)
    preallocated_X[:, 1:N] = X

    """
    Preallocate a covariance matrix of size d x capacity
    """
    preallocated_K = zeros(capacity, capacity)
    KXX = eval_KXX(ψ, X)
    preallocated_K[1:N, 1:N] = KXX + observation_noise * I

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
    preallocated_d = zeros(capacity)
    preallocated_d[1:N] = KXX \ y # should be using cholesky here

    preallocated_y = zeros(capacity)
    preallocated_y[1:N] = y

    observed = length(y)

    return GaussianProcess(
        ψ,
        preallocated_X,
        preallocated_K,
        preallocated_L,
        preallocated_y,
        preallocated_d,
        observation_noise,
        decision_rule,
        observed,
        capacity
    )
end

"""
When the kernel is changed, we need to update d, K, and L
"""
function set_kernel!(s::HybridSurrogate, kernel::RadialBasisFunction)
    @views begin
        # N = get_observed(s)
        # s.ψ = kernel
        # s.K[1:N, 1:N] .= eval_KXX(get_kernel(s), get_active_covariates(s), σn2=s.σn2)
        # s.L[1:N, 1:N] .= LowerTriangular(
        #     cholesky(
        #         Hermitian(s.K[1:N, 1:N])
        #     ).L
        # )
        # s.c[1:N] = s.L[1:N, 1:N]' \ (s.L[1:N, 1:N] \ get_active_observations(s))
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


function insert!(s::HybridSurrogate, x::Vector{T}, y::T) where T <: Real
    insert_index = get_observed(s) + 1
    s.X[:, insert_index] = x
    s.y[insert_index] = y
end

function update_covariance!(s::HybridSurrogate, x::Vector{T}, y::T) where T <: Real
    @views begin
        update_index = get_observed(s)
        active_X = get_covariates(s)[:, 1:update_index - 1]
        kernel = get_kernel(s)

        # Update the main diagonal
        s.K[update_index, update_index] = kernel(0.) + get_observation_noise(s)
        # Update the rows and columns with covariance vector formed from k(x, X)
        s.K[update_index, 1:update_index - 1] = eval_KxX(kernel, x, active_X)'
        s.K[1:update_index - 1, update_index] = s.K[update_index, 1:update_index - 1] 
    end
end

function update_cholesky!(s::HybridSurrogate)
    # Grab entries from update covariance matrix
    @views begin
        n = get_observed(s)
        B = s.K[n:n, 1:n-1]
        C = s.K[n:n, n:n]
        L = s.L[1:n-1, 1:n-1]
        
        # Compute the updated factorizations using schur complements
        L21 = B / L'
        L22 = cholesky(C - L21*L21').L

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
    observation_noise = get_observation_noise(s)
    y = get_active_observations(s)
    d, λ = coefficient_solve(KXX, PX, observation_noise, y)
    @views begin
        s.d[1:length(d)] = d
        s.λ[:] = λ
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
function condition!(s::HybridSurrogate, xnew::Vector{T}, ynew::T) where T <: Real
    if is_full(s) s = resize(s) end
    insert!(s, xnew, ynew) # Updates covariate matrix X and observation vector y
    increment!(s)
    update_covariance!(s, xnew, ynew) # Updates covariance matrix K
    update_cholesky!(s) # Updates cholesky factorization matrix L
    update_parametric_design_matrix!(s) # Updates parametric design matrix P
    update_coefficients!(s) # Updates coefficients d, λ
    return s
end

function eval(
    s::HybridSurrogate,
    x::Vector{T},
    θ::Vector{T}) where T <: Real
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
        sx.Hμ = function()
            H = zeros(dim, dim)
            # Reducing over the non-parametric component
            for j = 1:N
                H += d[j] * eval_Hk(kernel, x-X[:,j])
            end
            # Reducing over the parametric component
            HP = eval_Hbasis(parametric_basis, x)
            for j = 1:M
                H += λ[j] * HP[:, :, 1, j]
            end
            return H
        end

        # The variables v and w represent 
        sx.w = () -> (sx.px / P)' # P' \ sx.px'
        sx.v = () -> P \ (sx.kx - K*sx.w)
        sx.∇w = () -> (sx.∇px / P)'
        sx.∇v = () -> P \ (sx.∇kx - K*sx.∇w)
        sx.weighted_Hw = function()
            Hpx = eval_Hbasis(parametric_basis, x)
            weighted_Hw = zeros(dim, dim)

            for i in 1:dim
                for j in 1:dim
                    row_Hpx = reshape(Hpx[i, j, :, :], 1, length(parametric_basis))
                    Hw_ij = (row_Hpx / P)'
                    # Computes the mixed partial at indices i, j
                    weighted_Hw[i, j] = dot(sx.c0, Hw_ij)
                end
            end
            
            return weighted_Hw
        end

        # Reused terms c_i
        sx.c0 = () -> K*sx.w - sx.kx

        # Predictive standard deviation and its gradient and hessian
        sx.σ = function ()
            kxx = kernel(0)
            b = [sx.px sx.kx']
            return sqrt(kxx - dot(b, [sx.v; sx.w]))
        end
        sx.∇σ = () -> (sx.∇w'*sx.c0 - sx.w'*sx.∇kx') / sx.σ
        sx.Hσ = function()
            H = -sx.∇σ * sx.∇σ' - sx.∇w' * sx.∇kx' + sx.∇w'*(K*sx.∇w - sx.∇kx')
            w = sx.w

            for j in 1:length(w)
                H -= w[j] * eval_Hk(kernel, x - X[:, j])
            end

            H += sx.weighted_Hw
            H /= sx.σ

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
        sx.Hαx = () -> sx.d2g_dμ*sx.∇μ*sx.∇μ' + sx.dg_dμ*sx.Hμ + sx.d2g_dσ*sx.∇σ*sx.∇σ' + sx.dg_dσ*sx.Hσ
       
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


(s::HybridSurrogate)(x, θ) = eval(s, x, θ)
eval(sx) = sx.αxθ
gradient(sx; wrt_hypers=false) = wrt_hypers ? sx.∇αθ : sx.∇αx
hessian(sx; wrt_hypers=false) = wrt_hypers ? sx.Hαθ : sx.Hαx
mixed_partials(sx) = sx.d2α_dxdθ


function gp_draw(
    s::AS,
    xloc::Vector{T},
    θ::Vector{T};
    stdnormal::Union{Vector{T}, T},
    with_gradient::Bool = false,
    fantasy_index::Union{Int64, Nothing} = nothing) where {T <: Real, AS <: AbstractSurrogate}
    # We can actually embed this logic directly into the evaluation of the surrogate at some arbitrary location
    dim = length(xloc)

    if isnothing(fantasy_index)
        sx = s(xloc, θ)
    else
        sx = s(xloc, θ, fantasy_index=fantasy_index)
    end

    if with_gradient
        @assert length(stdnormal) == dim + 1 "stdnormal has dim = $(length(stdnormal)) but observation vector has dim = $(length(xloc))"
        return sx.dμ + sx.dσ * stdnormal
    else
        @assert length(stdnormal) == 1 "Function observation expects a scalar gaussian random number"
        return sx.μ + sx.σ * stdnormal
    end
end

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

    # Do the log determinant stuff here
    A = [zeros(m, m) P';
         P           K]

    return -dot(yz, dλ)/2 - n*log(2π)/2 - log(det(A))
end

function δlog_likelihood(s::HybridSurrogate, δθ::Vector{T}) where T <: Real
    kernel = get_kernel(s)
    X = get_active_covariates(s)
    δK = eval_Dθ_KXX(kernel, X, δθ)
    c = get_active_coefficients(s)
    L = get_active_cholesky(s)
    return (c'*δK*c - tr(L'\(L\δK)))/2
end

function ∇log_likelihood(s::HybridSurrogate)
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
    s::HybridSurrogate;
    lowerbounds::Vector{T},
    upperbounds::Vector{T},
    optim_options = Optim.Options(iterations=30)) where T <: Real

    function fg!(F, G, θ::Vector{T}) where T <: Real
        set_kernel!(s, set_hyperparameters!(get_kernel(s), θ))
        if G !== nothing G .= -∇log_likelihood(s) end
        if F !== nothing return -log_likelihood(s) end
    end

    res = optimize(
        Optim.only_fg!(fg!),
        lowerbounds,
        upperbounds,
        get_hyperparameters(get_kernel(s)),
        Fminbox(LBFGS()),
        optim_options
    )
    θ = Optim.minimizer(res)
    set_kernel!(s, set_hyperparameters!(get_kernel(s), θ))

    return nothing
end

Distributions.mean(sx) = sx.μ
Distributions.std(sx) = sqrt(sx.σ)