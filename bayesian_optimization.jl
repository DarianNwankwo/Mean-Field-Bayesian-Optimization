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
using StaticArrays
using SpecialFunctions
using NLopt
using TimerOutputs
using ElasticPDMats
using DiffResults
using BayesianOptimization

const to = TimerOutput()

include("constants.jl")
include("types.jl")
include("testfns.jl")
include("containers.jl")
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
    budget::Int,
    xnext::Vector{Float64},
    inner_optimizer_restarts::Int = 256,
    hyperparameter_optimizer_restarts::Int = 32,
    ) where T <: Real
    # Preallocate the vector xnext for solves of acquisition function
    cache = SurrogateEvaluationCache(length(spatial_lbs))

    # println("Performing Bayesian Optimization for $budget Iterations")
    print("Progress: ")
    for i in 1:budget
        print("|")
        setparams!(decision_rule, surrogate)
        @timeit to "Multistart Acquisition Solve" multistart_base_solve!(
            surrogate,
            decision_rule,
            xnext,
            spatial_lbs,
            spatial_ubs,
            cache,
            inner_optimizer_restarts
        )
        invalidate!(cache)
        ynext = testfn(xnext) + get_observation_noise(surrogate)
        surrogate = condition!(surrogate, xnext, ynext)
        print("-")
        optimize!(
            surrogate,
            lowerbounds=kernel_lbs,
            upperbounds=kernel_ubs,
            restarts=hyperparameter_optimizer_restarts,
        )
    end
    println()

    return surrogate
end