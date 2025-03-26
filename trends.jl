struct PolynomialTrend
    ϕ::PolynomialBasisFunction
    coefficients::Vector{Float64}
end


# Want to evaluate the polynomial at some arbitrary location and use this mechanism
# to adds arbitrary trends to a TestFunction
function (pt::PolynomialTrend)(x::Vector{T}) where T <: Real
    return dot(eval_basis(pt.ϕ, x), pt.coefficients)
end

function gradient(pt::PolynomialTrend, x::Vector{T}) where T <: Real
    Px = eval_∇basis(pt.ϕ, x)
    return Px * pt.coefficients
end

# Projection function: clips each component to the corresponding bounds.
function project(x, lower_bounds, upper_bounds)
    return clamp.(x, lower_bounds, upper_bounds)
end

# Backtracking line search to satisfy an Armijo condition.
function backtracking_line_search(f, g, x, d, lower_bounds, upper_bounds; α=1.0, β=0.5, c=1e-4)
    f_x = f(x)
    # Compute dot product once; note that d = -grad, so dot(g, d) is negative.
    gd = dot(g, d)
    while true
        # Compute the candidate step and project onto the bounds.
        x_new = project(x + α * d, lower_bounds, upper_bounds)
        if f(x_new) <= f_x + c * α * gd
            break
        end
        α *= β
        # Avoid excessively small step sizes.
        if α < 1e-10
            break
        end
    end
    return α
end

# Projected gradient descent solver.
#
# Arguments:
#   f             : function to minimize.
#   g!            : in-place function that computes the gradient at x.
#   x0            : initial guess (vector).
#   lower_bounds  : vector of lower bounds.
#   upper_bounds  : vector of upper bounds.
#   tol           : tolerance for convergence based on the projected gradient norm.
#   max_iter      : maximum number of iterations.
#
# Returns:
#   x, f(x)       : the computed minimizer and the corresponding function value.
function projected_gradient_descent(f, g!, x0, lower_bounds, upper_bounds; tol=1e-3, max_iter=1000)
    x = copy(x0)
    n = length(x)
    grad = zeros(n)
    
    for iter in 1:max_iter
        # Evaluate the gradient.
        g!(grad, x)
        
        # Compute the projected gradient: the difference between x and its projection after a full gradient step.
        projected_grad = x - project(x - grad, lower_bounds, upper_bounds)
        
        # Check for convergence using the norm of the projected gradient.
        if norm(projected_grad) < tol
            # println("Converged in $iter iterations.")
            return x, f(x)
        end
        
        # Descent direction is the negative gradient.
        d = -grad
        
        # Determine step size using backtracking line search.
        α = backtracking_line_search(f, grad, x, d, lower_bounds, upper_bounds)
        
        # Update and project the new iterate.
        x = project(x + α * d, lower_bounds, upper_bounds)
    end
    
    # println("Reached maximum iterations.")
    return x, f(x)
end

function +(testfn::TestFunction, pt::PolynomialTrend)
    new_f = (x) -> testfn.f(x) + pt(x)
    new_∇f = (x) -> testfn.∇f(x) + gradient(pt, x)

    # Functions for optim.jl
    f(x) = new_f(x)
    function g!(G, x)
        G[:] = new_∇f(x)
    end

    # The global minimizer is subject to shifting, so we find the global minimizer
    # algorithmically.
    M = 512
    lbs, ubs = get_bounds(testfn)
    initial_xs = randsample(M, testfn.dim, lbs, ubs)
    # inner_optimizer = LBFGS()
    minimizers = Vector{Vector{Float64}}(undef, M + 1)
    f_minimums = Vector{Float64}(undef, M + 1)

    
    # I should just used the projected gradient function here too
    candidates = []
    @sync @threads for i in 1:M
        # print("$i-")
        # results = optimize(
        #     f,
        #     g!,
        #     lbs,
        #     ubs,
        #     initial_xs[:, i],
        #     Fminbox(inner_optimizer),
        #     Optim.Options(x_tol=1e-3, f_tol=1e-3)
        # )
        minimizer, f_minimum = projected_gradient_descent(f, g!, initial_xs[:, i], lbs, ubs)
        minimizers[i] = minimizer
        f_minimums[i] = f_minimum
        # push!(candidates, (Optim.minimizer(results), Optim.minimum(results)))
    end
    candidates = [(minimizers[i], f_minimums[i]) for i in 1:M]
    push!(candidates, (testfn.xopt[1], testfn(testfn.xopt[1])))

    candidates = filter(pair -> !any(isnan.(pair[1])), candidates)
    mini, j_mini = findmin(pair -> pair[2], candidates)
    xopt = candidates[j_mini][1]
    
    return TestFunction(
        testfn.dim,
        testfn.bounds,
        [xopt],
        new_f,
        new_∇f
    )
end