using Plots
using Sobol
using Distributions 
using LinearAlgebra
using Optim
using ForwardDiff
using Statistics
using SharedArrays
using Roots
using FastGaussQuadrature
using IterTools
using Tables
using CSV
using DataFrames
using Base.Threads
using StaticArrays


include("constants.jl")
include("types.jl")
include("testfns.jl")
include("containers.jl")
include("lazy_struct.jl")
include("radial_basis_functions.jl")
include("polynomial_basis_functions.jl")
include("trends.jl")
include("decision_rules.jl")
include("surrogates.jl")
include("rbf_optim.jl")
include("utils.jl")


function bayesian_optimize!(
    surrogate::AbstractSurrogate,
    testfn::TestFunction,
    spatial_lbs::AbstractVector{T},
    spatial_ubs::AbstractVector{T},
    kernel_lbs::AbstractVector{T},
    kernel_ubs::AbstractVector{T},
    decision_rule_hyperparameters::AbstractVector{T},
    inner_optimizer_starts::AbstractMatrix{T},
    hyperparameter_optimizer_starts::AbstractMatrix{T},
    budget::Int) where T <: Real

    # Preallocate the vector xnext for solves of acquisition function
    xnext = zeros(Float64, testfn.dim)
    M = size(inner_optimizer_starts, 2)
    S = size(hyperparameter_optimizer_starts, 2)
    minimizers = Vector{Vector{Float64}}(undef, M)
    f_minimums = Vector{Float64}(undef, M)
    hyper_minimizers = Vector{Vector{Float64}}(undef, S)
    hyper_minimums = Vector{Float64}(undef, S)

    println("Performing Bayesian Optimization for $budget Iterations")
    print("Progress: ")
    for i in 1:budget
        print("|")
        # multistart_base_solve_threaded!(
            multistart_base_solve!(
            surrogate,
            xnext,
            spatial_lbs=spatial_lbs,
            spatial_ubs=spatial_ubs,
            guesses=inner_optimizer_starts,
            θfixed=decision_rule_hyperparameters,
            minimizers_container=minimizers,
            minimums_container=f_minimums
        )
        ynext = testfn(xnext) + get_observation_noise(surrogate)
        surrogate = condition!(surrogate, xnext, ynext)
        # print("-")
        optimize!(
            surrogate,
            lowerbounds=kernel_lbs,
            upperbounds=kernel_ubs,
            starts=hyperparameter_optimizer_starts,
            minimizers_container=hyper_minimizers,
            minimums_container=hyper_minimums
        )

    end
    # println()

    return surrogate
end