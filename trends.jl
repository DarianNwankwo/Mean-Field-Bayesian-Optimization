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
    M = 128
    lbs, ubs = get_bounds(testfn)
    initial_xs = randsample(M, testfn.dim, lbs, ubs)
    inner_optimizer = GradientDescent()
    
    candidates = []
    for i in 1:M
        results = optimize(f, g!, lbs, ubs, initial_xs[:, i], Fminbox(inner_optimizer))
        push!(candidates, (Optim.minimizer(results), Optim.minimum(results)))
    end
    
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