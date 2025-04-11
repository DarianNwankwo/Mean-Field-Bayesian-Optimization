# TODO: Add an output container for efficient evaluation of trend. Much
# like we have in our Surrogate struct.

struct PolynomialTrend
    ϕ::PolynomialBasisFunction
    coefficients::Vector{Float64}
    containers::PreallocatedContainers
end

function PolynomialTrend(ϕ::PolynomialBasisFunction, coefficients::Vector{T}, dim::Int64) where T <: Real
    NOT_IMPORTANT = 1
    return PolynomialTrend(
        ϕ,
        coefficients,
        PreallocatedContainers(length(ϕ), dim, NOT_IMPORTANT, NOT_IMPORTANT)
    )
end


# Want to evaluate the polynomial at some arbitrary location and use this mechanism
# to adds arbitrary trends to a TestFunction
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

# ------------------------------------------------------------------------
# Redefine CompositeData as CompositeFunction.
# This container holds a TestFunction and a PolynomialTrend.
struct CompositeFunction
    tf::TestFunction  # You may want to make this abstract or parametric if needed.
    pt::PolynomialTrend
end

# The composite function evaluation: add the test function and the trend.
function composite_f(x, cf::CompositeFunction)
    return cf.tf.f(x) + cf.pt(x)
end

# The composite gradient: compute the gradient of the test function and add the gradient of the trend.
function composite_∇f!(grad, x, cf::CompositeFunction)
    cf.tf.∇f!(grad, x)
    grad .+= gradient(cf.pt, x)
    return grad
end

# ------------------------------------------------------------------------
# Define a barrier type that holds the composite function along with fixed function pointers.
struct TestFunctionBarrier
    data::CompositeFunction
    fptr::Function   # Fixed pointer to composite_f
    ∇fptr::Function  # Fixed pointer to composite_∇f!
end

# Make the barrier callable (so it can be used in place of a plain function).
function (tfb::TestFunctionBarrier)(x)
    return tfb.fptr(x, tfb.data)
end

# Also define a wrapper for computing the gradient.
function ∇f!(grad, x, tfb::TestFunctionBarrier)
    return tfb.∇fptr(grad, x, tfb.data)
end

# function +(testfn::TestFunction, pt::PolynomialTrend)
#     # grad = zeros(testfn.dim)
#     function new_f(x)
#         term1 = testfn.f(x)
#         term2 = pt(x)
#         return term1 + term2
#     end

#     function new_∇f!(grad, x)
#         return testfn.∇f!(grad, x) + gradient(pt, x)
#     end

#     # The global minimizer is subject to shifting, so we find the global minimizer
#     # algorithmically.
#     M = 64
#     lbs, ubs = get_bounds(testfn)
#     initial_xs = randsample(M, testfn.dim, lbs, ubs)
#     inner_optimizer = LBFGS()
#     minimizers = Vector{Vector{Float64}}(undef, M + 1)
#     f_minimums = Vector{Float64}(undef, M + 1)

    
#     # I should just used the projected gradient function here too
#     candidates = []
#     # @sync @threads for i in 1:M
#     for i in 1:M
#         # print("$i-")
#         results = optimize(
#             new_f,
#             new_∇f!,
#             lbs,
#             ubs,
#             initial_xs[:, i],
#             Fminbox(inner_optimizer),
#             Optim.Options(x_tol=1e-3, f_tol=1e-3, time_limit=.1)
#         )
#         # minimizer, f_minimum = projected_gradient_descent(f, g!, initial_xs[:, i], lbs, ubs)
#         minimizers[i] = Optim.minimizer(results)
#         f_minimums[i] = Optim.minimum(results)
#     end
#     candidates = [(minimizers[i], f_minimums[i]) for i in 1:M]
#     push!(candidates, (testfn.xopt[1], testfn(testfn.xopt[1])))

#     candidates = filter(pair -> !any(isnan.(pair[1])), candidates)
#     mini, j_mini = findmin(pair -> pair[2], candidates)
#     xopt = candidates[j_mini][1]
    
#     return TestFunction(
#         testfn.dim,
#         testfn.bounds,
#         Tuple([xopt]),
#         new_f,
#         new_∇f!
#     )
# end
# ------------------------------------------------------------------------
# Finally, create a composition function to combine an arbitrary TestFunction and PolynomialTrend.
# This serves as a replacement for operator+.
function plus(
    tf::TestFunction,
    pt::PolynomialTrend,
    initial_xs::Matrix{Float64},
    minimizers::Vector{Vector{Float64}},
    f_minimums::Vector{Float64}
    )
    cf = CompositeFunction(tf, pt)
    barrier = TestFunctionBarrier(cf, composite_f, composite_∇f!)
    ∇barrier! = (grad, x) -> ∇f!(grad, x, barrier)

    # The global minimizer is subject to shifting, so we find the global minimizer
    # algorithmically.
    lbs, ubs = get_bounds(tf)
    M = length(minimizers)

    for i in 1:M-1
        # print("$i-")
        results = optimize(
            barrier,
            ∇barrier!,
            lbs,
            ubs,
            initial_xs[:, i],
            Fminbox(LBFGS()),
            Optim.Options(x_tol=1e-3, f_tol=1e-3, time_limit=.1)
        )
        minimizers[i] = Optim.minimizer(results)
        f_minimums[i] = Optim.minimum(results)
    end
    minimizers[M] = tf.xopt[1]
    f_minimums[M] = tf(tf.xopt[1])
    idx = argmin(f_minimums)
    minimizer = minimizers[idx]

    return TestFunction(
        tf.dim,
        tf.bounds,
        Tuple([minimizer]),
        barrier,
        ∇barrier!
    )
end