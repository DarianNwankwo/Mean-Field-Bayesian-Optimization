struct PreallocatedContainers
    px::Matrix{Float64} # Fixed size
    grad_px::Matrix{Float64} # Fixed size
    Hpx::Matrix{Float64} # Fixed size
    grad_k::Vector{Float64} # Fixed size
    Hk::Matrix{Float64} # Fixed size
    KxX::Vector{Float64} # Variable size
    grad_KxX::Matrix{Float64} # Variable size
    A::Matrix{Float64} # Variable size
    v::Vector{Float64} # Variable size
    grad_v::Matrix{Float64} # Variable size
    w::Vector{Float64} # Variable size
    grad_w::Matrix{Float64} # Variable size
    diff_x::Vector{Float64} # Fixed size
    δdiff_x::Vector{Float64}
    δθ::Vector{Float64}
    grad_L::Vector{Float64}
    δKXX::Matrix{Float64}
    Hσ::Matrix{Float64} # Fixed size
    δψij::Vector{Float64}
    Hf::Matrix{Float64} # Fixed sized intermediate container
    Hz::Matrix{Float64}
    zz::Matrix{Float64} # Fixed size
end

function PreallocatedContainers(
    num_of_basis_funcitons::Int,
    dim::Int,
    max_obs::Int,
    hypers_length::Int)
    m = num_of_basis_funcitons
    return PreallocatedContainers(
        zeros(Float64, 1, m),
        zeros(Float64, dim, m),
        zeros(Float64, dim, dim),
        zeros(Float64, dim),
        zeros(Float64, dim, dim),
        zeros(Float64, max_obs),
        zeros(Float64, dim, max_obs),
        zeros(Float64, m + max_obs, m + max_obs),
        zeros(Float64, m + max_obs),
        zeros(Float64, dim, m + max_obs),
        zeros(Float64, m + max_obs),
        zeros(Float64, m + max_obs, dim),
        zeros(Float64, dim),
        zeros(Float64, dim),
        zeros(Float64, hypers_length),
        zeros(Float64, hypers_length),
        zeros(Float64, max_obs, max_obs),
        zeros(Float64, dim, dim),
        zeros(Float64, hypers_length),
        zeros(Float64, dim, dim),
        zeros(Float64, dim, dim),
        zeros(Float64, m, m)
    )
end