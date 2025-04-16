function base_solve(
    surrogate::AbstractSurrogate,
    decision_rule::AbstractDecisionRule,
    spatial_lbs::AbstractVector{T},
    spatial_ubs::AbstractVector{T},
    xstart::AbstractVector{T},
    cache::SurrogateEvaluationCache) where T <: Real
    fun(x) = -eval(surrogate, decision_rule, x, cache)
    fun_grad!(g, x) = begin
        g[:] = -eval_gradient(surrogate, decision_rule, x, cache) 
    end
    fun_hess!(h, x) = begin
        h[:, :] = -eval_hessian(surrogate, decision_rule, x, cache)
    end

    df = TwiceDifferentiable(fun, fun_grad!, fun_hess!, xstart)
    dfc = TwiceDifferentiableConstraints(spatial_lbs, spatial_ubs)
    res = optimize(
        df, dfc, xstart, IPNewton(),
        Optim.Options(
            x_tol=1e-4,
            f_tol=1e-4,
            time_limit=NEWTON_SOLVE_TIME_LIMIT,
            outer_iterations=100,
            # iterations=20,
        )
    )
    
    return Optim.minimizer(res), res
end

function base_solve_nlopt(
    surrogate::AbstractSurrogate,
    decision_rule::AbstractDecisionRule,
    spatial_lbs::AbstractVector{T},
    spatial_ubs::AbstractVector{T},
    xstart::AbstractVector{T},
    cache::SurrogateEvaluationCache
) where T <: Real
    n = length(xstart)
    # opt = NLopt.Opt(:LD_SLSQP, n)
    opt = NLopt.Opt(:LD_LBFGS, n)
    
    # Set the lower and upper bounds
    lower_bounds!(opt, spatial_lbs)
    upper_bounds!(opt, spatial_ubs)
    
    # Set stopping criteria
    xtol_rel!(opt, 1e-4)
    ftol_rel!(opt, 1e-4)
    maxeval!(opt, 100)             # maximum number of evaluations/iterations
    # Optionally, set a time limit if NEWTON_SOLVE_TIME_LIMIT is defined:
    maxtime!(opt, NEWTON_SOLVE_TIME_LIMIT)
    
    # Define the objective callback.
    # NLopt callbacks accept (x, grad) where:
    # - x is the current iterate (Vector{T})
    # - grad is an output vector (if nonempty) to be filled with gradient information.
    function nlopt_obj(x, grad)
        if length(grad) > 0
            grad[:] = eval_gradient(surrogate, decision_rule, x, cache)
        end
        return eval(surrogate, decision_rule, x, cache)
    end
    
    # nlopt_obj = wrap_gradient(decision_rule_functor(decision_rule, surrogate))
    # Perform the optimization
    NLopt.max_objective!(opt, nlopt_obj)
    f_min, x_min, ret = NLopt.optimize(opt, xstart)
    
    # Return the minimizer and a tuple with more detailed results (x_min, function value, and termination code)
    # return x_min, (x_min, f_min, ret)
    return x_min, f_min
end


function multistart_base_solve!(
    surrogate::AbstractSurrogate,
    decision_rule::AbstractDecisionRule,
    xfinal::AbstractVector{T},
    spatial_lbs::AbstractVector{T},
    spatial_ubs::AbstractVector{T},
    guesses::Matrix{T},
    cache::SurrogateEvaluationCache,
    minimizers_container::Vector{Vector{T}},
    minimums_container::AbstractVector{T}) where T <: Real
    # if get_name(get_decision_rule(surrogate)) == "Random"
    #     xfinal[:] = spatial_lbs .+ (spatial_ubs .- spatial_lbs) .* rand(length(spatial_lbs))
    #     return nothing
    # end

    M = size(guesses, 2)    
    for i in 1:size(guesses, 2)
        minimizer, f_min = base_solve_nlopt(
            surrogate,
            decision_rule,
            spatial_lbs,
            spatial_ubs,
            guesses[:, i],
            cache
        )
        minimizers_container[i] = minimizer
        minimums_container[i] = f_min
    end
    
    idx = argmin(minimums_container)
    xfinal[:] = minimizers_container[idx]

    return nothing
end