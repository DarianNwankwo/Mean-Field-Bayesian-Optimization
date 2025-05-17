mutable struct SchurBuffer{T}
    Y::Matrix{T}  # size n by m where n = n_0 + budget: variable
    S::Matrix{T}  # size m by m: fixed
    tmp::Vector{T}  # size n: variable
    r::Vector{T}  # size m: fixed
    w1::Vector{T}  # size m: fixed
    w0::Vector{T}  # size n: variable
    w::Vector{T}  # size n + m: variable
    d::Vector{T}  # size n: variable
    λ::Vector{T}  # size m: fixed
    active_index::Int64
    valid::Bool
    x::Vector{T}
end
invalidate!(buffer::SchurBuffer) = buffer.valid = false

# TODO: Add resizing logic
function SchurBuffer{T}(max_obs, num_basis_functions, active_index, dim) where T
    return SchurBuffer{T}(
        zeros(T, max_obs, num_basis_functions),
        zeros(T, num_basis_functions, num_basis_functions),
        zeros(T, max_obs),
        zeros(T, num_basis_functions),
        zeros(T, num_basis_functions),
        zeros(T, max_obs),
        zeros(T, max_obs + num_basis_functions),
        zeros(T, max_obs),
        zeros(T, num_basis_functions),
        active_index,
        false,
        zeros(T, dim)
    )
end

get_Y(sb::SchurBuffer) = @view sb.Y[1:sb.active_index, :]
get_S(sb::SchurBuffer) = @view sb.S[:, :]
get_tmp(sb::SchurBuffer) = @view sb.tmp[1:sb.active_index]
get_r(sb::SchurBuffer) = @view sb.r[:]
get_w1(sb::SchurBuffer) = @view sb.w1[:]
get_w0(sb::SchurBuffer) = @view sb.w0[1:sb.active_index]
get_w(sb::SchurBuffer) = @view sb.w[1:sb.active_index + length(sb.w1)]
get_d(sb::SchurBuffer) = @view sb.d[1:sb.active_index]
get_λ(sb::SchurBuffer) = @view sb.λ[:]

struct PreallocatedContainers{T <: Real}
    px::Matrix{T} # Fixed size
    grad_px::Matrix{T} # Fixed size
    Hpx::Matrix{T} # Fixed size
    grad_k::Vector{T} # Fixed size
    Hk::Matrix{T} # Fixed size
    KxX::Vector{T} # Variable size
    grad_KxX::Matrix{T} # Variable size
    A::Matrix{T} # Variable size
    v::Vector{T} # Variable size
    grad_v::Matrix{T} # Variable size
    w::Vector{T} # Variable size
    grad_w::Matrix{T} # Variable size
    diff_x::Vector{T} # Fixed size
    δdiff_x::Vector{T}
    δθ::Vector{T}
    grad_L::Vector{T}
    δKXX::Matrix{T}
    Hσ::Matrix{T} # Fixed size
    δψij::Vector{T}
    Hf::Matrix{T} # Fixed sized intermediate container
    Hz::Matrix{T}
    zz::Matrix{T} # Fixed size
    chol_workspace::Matrix{T}
    yc::Vector{T}
    schur::SchurBuffer{T}
end

function PreallocatedContainers{T}(
    num_of_basis_funcitons::Int,
    dim::Int,
    max_obs::Int,
    hypers_length::Int,
    active_index::Int64) where T <: Real
    m = num_of_basis_funcitons
    schur_buffer = SchurBuffer{T}(max_obs, num_of_basis_funcitons, active_index, dim)
    return PreallocatedContainers{T}(
        zeros(T, 1, m),
        zeros(T, dim, m),
        zeros(T, dim, dim),
        zeros(T, dim),
        zeros(T, dim, dim),
        zeros(T, max_obs),
        zeros(T, dim, max_obs),
        zeros(T, m + max_obs, m + max_obs),
        zeros(T, m + max_obs),
        zeros(T, dim, m + max_obs),
        zeros(T, m + max_obs),
        zeros(T, m + max_obs, dim),
        zeros(T, dim),
        zeros(T, dim),
        zeros(T, hypers_length),
        zeros(T, hypers_length),
        zeros(T, max_obs, max_obs),
        zeros(T, dim, dim),
        zeros(T, hypers_length),
        zeros(T, dim, dim),
        zeros(T, dim, dim),
        zeros(T, m, m),
        zeros(T, max_obs, max_obs),
        zeros(T, max_obs),
        schur_buffer
    )
end

get_diff_x(pc::PreallocatedContainers{T}) where T <: Real = @view pc.diff_x[:]


struct HybridMomentsContainer{T <: Real}
    grad_μ::Vector{T}
    grad_σ::Vector{T}
end

function HybridMomentsContainer{T}(dim::Int) where T <: Real
    return HybridMomentsContainer(
        zeros(T, dim),
        zeros(T, dim)
    )
end