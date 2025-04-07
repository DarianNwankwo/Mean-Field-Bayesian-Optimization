# TODO: Add an output container for efficient evaluation of trend. Much
# like we have in our Surrogate struct.

struct PolynomialTrend
    ϕ::PolynomialBasisFunction
    coefficients::Vector{Float64}
    containers::PreallocatedContainers
end

function PolynomialTrend(ϕ::PolynomialBasisFunction, coefficients::Vector{T}, dim::Int64) where T <: Real
    return PolynomialTrend(ϕ, coefficients, PreallocatedContainers(ϕ, dim))
end


# Want to evaluate the polynomial at some arbitrary location and use this mechanism
# to adds arbitrary trends to a TestFunction
# function (pt::PolynomialTrend)(x::Vector{T}) where T <: Real
#     return dot(eval_basis(pt.ϕ, x), pt.coefficients)
# end

# function (pt::PolynomialTrend)(x::Vector{T}, out::AbstractMatrix{T}) where T <: Real
function (pt::PolynomialTrend)(x::AbstractVector{T}) where T <: Real
    # eval_basis!(pt.ϕ, x, (@view pt.containers.px[1, :]))
    eval_basis!(pt.ϕ, x, (@view pt.containers.px[1:1, :]))
    return dot(pt.containers.px, pt.coefficients)
end


function gradient(pt::PolynomialTrend, x::AbstractVector{T}) where T <: Real
    eval_∇basis!(
        pt.ϕ,
        x,
        (@view pt.containers.∇px[:, :])
    )
    # println("pt.containers.∇px = $(pt.containers.∇px)")
    return pt.containers.∇px[:, :] * pt.coefficients
end

function +(testfn::TestFunction, pt::PolynomialTrend)
    # grad = zeros(testfn.dim)
    new_f = (x) -> testfn.f(x) + pt(x)
    new_∇f! = (grad, x) -> testfn.∇f(grad, x) + gradient(pt, x)

    # Functions for optim.jl
    f(x) = new_f(x)
    function g!(G, x)
        # G[:] = new_∇f(x)
        new_∇f!(G, x)
    end

    # The global minimizer is subject to shifting, so we find the global minimizer
    # algorithmically.
    M = 64
    lbs, ubs = get_bounds(testfn)
    initial_xs = randsample(M, testfn.dim, lbs, ubs)
    inner_optimizer = LBFGS()
    minimizers = Vector{Vector{Float64}}(undef, M + 1)
    f_minimums = Vector{Float64}(undef, M + 1)

    
    # I should just used the projected gradient function here too
    candidates = []
    # @sync @threads for i in 1:M
    for i in 1:M
        # print("$i-")
        results = optimize(
            f,
            g!,
            lbs,
            ubs,
            initial_xs[:, i],
            Fminbox(inner_optimizer),
            Optim.Options(x_tol=1e-3, f_tol=1e-3, time_limit=.1)
        )
        # minimizer, f_minimum = projected_gradient_descent(f, g!, initial_xs[:, i], lbs, ubs)
        minimizers[i] = Optim.minimizer(results)
        f_minimums[i] = Optim.minimum(results)
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
        new_∇f!
    )
end