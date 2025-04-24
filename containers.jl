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
end

function PreallocatedContainers{T}(
    num_of_basis_funcitons::Int,
    dim::Int,
    max_obs::Int,
    hypers_length::Int) where T <: Real
    m = num_of_basis_funcitons
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
        zeros(T, max_obs, max_obs)
    )
end