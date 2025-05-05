# TODO: Add an output container for efficient evaluation of trend. Much
# like we have in our Surrogate struct.

struct PolynomialTrend
    ϕ::PolynomialBasisFunction
    coefficients::Vector{Float64}
    containers::PreallocatedContainers
end

function PolynomialTrend(ϕ::PolynomialBasisFunction, coefficients::Vector{T}, dim::Int64) where T
    NOT_IMPORTANT = 1
    return PolynomialTrend(
        ϕ,
        coefficients,
        PreallocatedContainers{Float64}(length(ϕ), dim, NOT_IMPORTANT, NOT_IMPORTANT, NOT_IMPORTANT)
    )
end


# Want to evaluate the polynomial at some arbitrary location and use this mechanism
# to adds arbitrary trends to a TestFunction
function (pt::PolynomialTrend)(x::AbstractVector{T}) where T
    # eval_basis!(pt.ϕ, x, (@view pt.containers.px[1, :]))
    eval_basis!(pt.ϕ, x, (@view pt.containers.px[1:1, :]))
    return dot(pt.containers.px, pt.coefficients)
end


function gradient(pt::PolynomialTrend, x::AbstractVector{T}) where T
    eval_∇basis!(
        pt.ϕ,
        x,
        (@view pt.containers.grad_px[:, :])
    )
    # println("pt.containers.∇px = $(pt.containers.∇px)")
    return pt.containers.grad_px[:, :] * pt.coefficients
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
    fptr::F where F <: Function   # Fixed pointer to composite_f
    ∇fptr::G where G <: Function # Fixed pointer to composite_∇f!
end

# Make the barrier callable (so it can be used in place of a plain function).
function (tfb::TestFunctionBarrier)(x)
    return tfb.fptr(x, tfb.data)
end

# Also define a wrapper for computing the gradient.
function ∇f!(grad, x, tfb::TestFunctionBarrier)
    return tfb.∇fptr(grad, x, tfb.data)
end


# ------------------------------------------------------------------------
# Finally, create a composition function to combine an arbitrary TestFunction and PolynomialTrend.
# This serves as a replacement for operator+.
function plus(
    tf::TestFunction,
    pt::PolynomialTrend;
    M::Int = 256
    )
    cf = CompositeFunction(tf, pt)
    barrier = TestFunctionBarrier(cf, composite_f, composite_∇f!)
    ∇barrier! = (grad, x) -> ∇f!(grad, x, barrier)

    # The global minimizer is subject to shifting, so we find the global minimizer
    # algorithmically.
    lbs, ubs = get_bounds(tf)
    minf = tf(tf.xopt[1])
    minx = tf.xopt[1]
    seq = ScaledLHSIterator(lbs, ubs, M)

    for xstart in seq
        # print("$i-")
        results = Optim.optimize(
            barrier,
            ∇barrier!,
            lbs,
            ubs,
            convert(Vector, xstart),
            Fminbox(LBFGS()),
            Optim.Options(x_tol=1e-3, f_tol=1e-3, time_limit=.1)
        )
        if Optim.minimum(results) < minf
            minf = Optim.minimum(results)
            minx = Optim.minimizer(results)
        end
    end

    return TestFunction(
        tf.dim,
        tf.bounds,
        Tuple([minx]),
        barrier,
        ∇barrier!
    )
end