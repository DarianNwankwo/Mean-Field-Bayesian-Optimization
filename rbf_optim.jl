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
        Optim.Options(
            x_tol=1e-3,
            f_tol=1e-3,
            time_limit=NEWTON_SOLVE_TIME_LIMIT,
            outer_iterations=20,
            iterations=20,
        )
    )
    
    return Optim.minimizer(res), res
end

# Projection operator: clips each component of x to lie within [lower, upper]
function project(x::AbstractVector, lower::AbstractVector, upper::AbstractVector)
    return clamp.(x, lower, upper)
end

# Backtracking line search that ensures sufficient decrease.
function line_search(f, x, p, g, lower, upper; α0=1.0, ρ=0.5, c=1e-4)
    α = α0
    while f(project(x .+ α .* p, lower, upper)) > f(x) + c * α * dot(g, p)
        α *= ρ
        # If α becomes too small, break out (could also return a failure flag)
        if α < 1e-8
            break
        end
    end
    return α
end

# Projected Newton method for minimizing f subject to box constraints.
function projected_newton(f, grad, hess, x0, lower, upper; tol=1e-3, max_iter=100)
    x = project(x0, lower, upper)
    for iter = 1:max_iter
        g = grad(x)
        H = hess(x)
        
        # Solve H * p = -g, but if the Hessian is nearly singular or not PD,
        # fall back to the gradient direction.
        try
            p = -H \ g
        catch
            p = -g
        end

        # Check if the computed step is a descent direction
        if dot(g, p) > 0
            p = -g
        end
        
        p = vec(p)
        # Determine a suitable step length
        α = line_search(f, x, p, g, lower, upper)
        
        # Update the iterate with projection onto the feasible region.
        x_new = project(x .+ α .* p, lower, upper)
        
        # Check for convergence
        if norm(x_new - x) < tol
            return x_new, f(x_new) #iter
        end
        
        x = x_new
    end
    return x, f(x) # max_iter
end

function base_solve_alt(
    surrogate::AbstractSurrogate;
    spatial_lbs::Vector{T},
    spatial_ubs::Vector{T},
    xstart::Vector{T},
    θfixed::Vector{T}) where T <: Real

    fun(x) = -eval(surrogate(x, θfixed))
    grad(x) = -gradient(surrogate(x, θfixed))
    hess(x) = -hessian(surrogate(x, θfixed))

    minimizer, f_minimum = projected_newton(fun, grad, hess, xstart, spatial_lbs, spatial_ubs)
    
    return minimizer, f_minimum
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
        # print("$i-")
        xi = guesses[:, i]

        # minimizer, res = base_solve(
        #     surrogate,
        #     spatial_lbs=spatial_lbs,
        #     spatial_ubs=spatial_ubs,
        #     xstart=xi,
        #     θfixed=θfixed
        # )
        # push!(candidates, (minimizer, minimum(res)))
        minimizer, f_minimum = base_solve_alt(
            surrogate,
            spatial_lbs=spatial_lbs,
            spatial_ubs=spatial_ubs,
            xstart=xi,
            θfixed=θfixed
        )
        push!(candidates, (minimizer, f_minimum))
    end
    
    candidates = filter(pair -> !any(isnan.(pair[1])), candidates)
    mini, j_mini = findmin(pair -> pair[2], candidates)
    xfinal .= candidates[j_mini][1]

    return nothing
end

function multistart_base_solve_threaded!(
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
    nstarts = size(guesses, 2)
    candidates = []
    minimizers = Vector{Vector{Float64}}(undef, nstarts)
    f_minimums = Vector{Float64}(undef, nstarts)
    
    @sync @threads for i in 1:nstarts
        # print("$i-")
        xi = guesses[:, i]
        
        minimizer, f_minimum = base_solve_alt(
            surrogate,
            spatial_lbs=spatial_lbs,
            spatial_ubs=spatial_ubs,
            xstart=xi,
            θfixed=θfixed
        )
        minimizers[i] = minimizer
        f_minimums[i] = f_minimum
    end
    
    candidates = [(minimizers[i], f_minimums[i]) for i in 1:nstarts]
    candidates = filter(pair -> !any(isnan.(pair[1])), candidates)
    mini, j_mini = findmin(pair -> pair[2], candidates)
    xfinal .= candidates[j_mini][1]

    return nothing
end