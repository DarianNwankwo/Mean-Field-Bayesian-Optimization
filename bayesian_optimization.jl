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
using SpecialFunctions
using NLopt
using TimerOutputs

const to = TimerOutput()

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
    decision_rule::AbstractDecisionRule,
    testfn::TestFunction,
    spatial_lbs::AbstractVector{T},
    spatial_ubs::AbstractVector{T},
    kernel_lbs::AbstractVector{T},
    kernel_ubs::AbstractVector{T},
    inner_optimizer_starts::AbstractMatrix{T},
    hyperparameter_optimizer_starts::AbstractMatrix{T},
    budget::Int,
    minimizers::Vector{Vector{Float64}},
    f_minimums::Vector{Float64},
    hyper_minimizers::Vector{Vector{Float64}},
    hyper_minimums::Vector{Float64},
    xnext::Vector{Float64}
    ) where T <: Real
    # Preallocate the vector xnext for solves of acquisition function
    cache = SurrogateEvaluationCache(length(spatial_lbs))
    M = size(inner_optimizer_starts, 2)
    S = size(hyperparameter_optimizer_starts, 2)

    # println("Performing Bayesian Optimization for $budget Iterations")
    print("Progress: ")
    for i in 1:budget
        print("|")
        # multistart_base_solve_threaded!(
        @timeit to "Multistart Acquisition Solve" multistart_base_solve!(
            surrogate,
            decision_rule,
            xnext,
            spatial_lbs,
            spatial_ubs,
            inner_optimizer_starts,
            cache,
            minimizers,
            f_minimums
        )
        invalidate!(cache)
        ynext = testfn(xnext) + get_observation_noise(surrogate)
        @timeit to "Model Update" surrogate = condition!(surrogate, xnext, ynext)
        # print("-")
        @timeit to "Hyperparameter Optimization" optimize!(
            surrogate,
            lowerbounds=kernel_lbs,
            upperbounds=kernel_ubs,
            starts=hyperparameter_optimizer_starts,
            minimizers_container=hyper_minimizers,
            minimums_container=hyper_minimums
        )
    end
    println()

    return surrogate
end