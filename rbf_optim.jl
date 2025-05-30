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
    xtol_rel!(opt, X_RELTOL)
    ftol_rel!(opt, F_RELTOL)
    maxeval!(opt, 100)             # maximum number of evaluations/iterations
    # Optionally, set a time limit if NEWTON_SOLVE_TIME_LIMIT is defined:
    maxtime!(opt, NEWTON_SOLVE_TIME_LIMIT)
    
    # Define the objective callback.
    # NLopt callbacks accept (x, grad) where:
    # - x is the current iterate (Vector{T})
    # - grad is an output vector (if nonempty) to be filled with gradient information.
    function nlopt_obj(x, grad)
        if length(grad) > 0
            grad .= eval_gradient(surrogate, decision_rule, x, cache)
        end
        return eval(surrogate, decision_rule, x, cache)
    end
    
    # nlopt_obj = wrap_gradient(decision_rule_functor(decision_rule, surrogate))
    # Perform the optimization
    NLopt.max_objective!(opt, nlopt_obj)
    f_min, x_min, ret = NLopt.optimize(opt, xstart)
    
    # Return the minimizer and a tuple with more detailed results (x_min, function value, and termination code)
    return x_min, f_min
end


function multistart_base_solve!(
    surrogate::AbstractSurrogate,
    decision_rule::AbstractDecisionRule,
    xfinal::AbstractVector{T},
    spatial_lbs::AbstractVector{T},
    spatial_ubs::AbstractVector{T},
    cache::SurrogateEvaluationCache,
    restarts::Int = 256) where T <: Real
    if get_name(decision_rule) == RANDOM_SAMPLER_NAME
        xfinal[:] .= randsample(1, length(xfinal), spatial_lbs, spatial_ubs)
        return nothing
    end

    maxf = -Inf
    maxx = spatial_lbs
    seq = ScaledLHSIterator(spatial_lbs, spatial_ubs, restarts)
    for xstart in seq
        maximizer, f_max = base_solve_nlopt(
            surrogate,
            decision_rule,
            spatial_lbs,
            spatial_ubs,
            convert(Vector, xstart),
            cache
        )
        invalidate!(cache)
        if f_max > maxf
            maxf = f_max
            maxx = maximizer
        end
    end
    xfinal[:] .= maxx
    return nothing
end