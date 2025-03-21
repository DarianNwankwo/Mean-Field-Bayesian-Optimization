function base_solve(
    surrogate::AbstractSurrogate;
    spatial_lbs::Vector{T},
    spatial_ubs::Vector{T},
    xstart::Vector{T},
    θfixed::Vector{T}) where T <: Real

    function fun(x)
        surrogate_at_xθ = surrogate(x, θfixed)
        return -eval(surrogate_at_xθ)
    end
    
    function fun_grad!(g, x)
        surrogate_at_xθ = surrogate(x, θfixed)
        g[:] = -gradient(surrogate_at_xθ)
    end
    
    function fun_hess!(h, x)
        surrogate_at_xθ = surrogate(x, θfixed)

        h[:, :] .= -hessian(surrogate_at_xθ)
    end

    df = TwiceDifferentiable(fun, fun_grad!, fun_hess!, xstart)
    dfc = TwiceDifferentiableConstraints(spatial_lbs, spatial_ubs)
    res = optimize(
        df, dfc, xstart, IPNewton(),
        Optim.Options(x_tol=1e-3, f_tol=1e-3, time_limit=NEWTON_SOLVE_TIME_LIMIT)
    )
    
    return Optim.minimizer(res), res
end

function multistart_base_solve!(
    surrogate::AbstractSurrogate,
    xfinal::Vector{T};
    spatial_lbs::Vector{T},
    spatial_ubs::Vector{T},
    guesses::Matrix{T},
    θfixed::Vector{T}) where T <: Real
    if get_name(get_decision_rule(surrogate)) == "Random"
        xfinal[:] = spatial_lbs .+ (spatial_ubs .- spatial_lbs) .* rand(length(spatial_lbs))
        return nothing
    end
    candidates = []
    
    for i in 1:size(guesses, 2)
        xi = guesses[:, i]

        minimizer, res = base_solve(
            surrogate,
            spatial_lbs=spatial_lbs,
            spatial_ubs=spatial_ubs,
            xstart=xi,
            θfixed=θfixed
        )
        push!(candidates, (minimizer, minimum(res)))
    end
    
    candidates = filter(pair -> !any(isnan.(pair[1])), candidates)
    mini, j_mini = findmin(pair -> pair[2], candidates)
    xfinal .= candidates[j_mini][1]

    return nothing
end