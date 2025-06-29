# TODO: The types need to be fully exposed when specifying the type of the attributes
# on our structs.
# TODO: Add gradient container to functions

# struct PolynomialTrend
#     ϕ::PolynomialBasisFunction
#     coefficients::Vector{Float64}
#     containers::PreallocatedContainers
# end
struct PolynomialTrend{N, BF<:NTuple{N,Function}, BG<:NTuple{N,Function}, T}
    ϕ::PolynomialBasisFunction{N,BF,BG}
    coefficients::Vector{T}
    containers::PreallocatedContainers{T}
end

# function PolynomialTrend(ϕ::PolynomialBasisFunction, coefficients::Vector{T}, dim::Int64) where T
#     NOT_IMPORTANT = 1
#     return PolynomialTrend(
#         ϕ,
#         coefficients,
#         PreallocatedContainers{Float64}(length(ϕ), dim, NOT_IMPORTANT, NOT_IMPORTANT, NOT_IMPORTANT)
#     )
# end
function PolynomialTrend(
    ϕ::PolynomialBasisFunction{N,BF,BG}, coefficients::Vector{T}, dim::Int
    ) where {N,BF<:NTuple{N,Function},BG<:NTuple{N,Function},T<:Real}
    # choose container sizes as before
    NOT_IMPORTANT = 1
    return PolynomialTrend{N,BF,BG,T}(
        ϕ,
        coefficients,
        PreallocatedContainers{T}(length(ϕ), dim, NOT_IMPORTANT, NOT_IMPORTANT, NOT_IMPORTANT)
    )
end


# Want to evaluate the polynomial at some arbitrary location and use this mechanism
# to adds arbitrary trends to a TestFunction
function (pt::PolynomialTrend)(x::AbstractVector{T}) where T
    # eval_basis!(pt.ϕ, x, (@view pt.containers.px[1, :]))
    eval_basis!(pt.ϕ, x, (@view pt.containers.px[1:1, :]))
    return dot(pt.containers.px, pt.coefficients)
end

function gradient(
    # grad::AbstractVector{T},
    pt::PolynomialTrend{N,BF,BG,T},
    x::AbstractVector{T}) where {N,BF<:NTuple{N,Function}, BG<:NTuple{N,Function}, T <: Real}
# function gradient(pt::PolynomialTrend, x::AbstractVector{T}) where T
    eval_∇basis!(
        pt.ϕ,
        x,
        pt.containers.grad_px
    )
    # mul!(grad, pt.containers.grad_px, pt.coefficients)
    # return grad
    return pt.containers.grad_px * pt.coefficients
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
# struct TestFunctionBarrier
#     data::CompositeFunction
#     fptr::F where F <: Function   # Fixed pointer to composite_f
#     ∇fptr::G where G <: Function # Fixed pointer to composite_∇f!
# end
struct TestFunctionBarrier{F <: Function, G <: Function}
    data::CompositeFunction
    fptr::F
    ∇fptr::G
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
    M::Int64 = 4096
    )
    cf = CompositeFunction(tf, pt)
    barrier = TestFunctionBarrier(cf, composite_f, composite_∇f!)
    ∇barrier! = (grad, x) -> ∇f!(grad, x, barrier)

    # The global minimizer is subject to shifting, so we find the global minimizer
    # algorithmically.
    lbs, ubs = get_bounds(tf)
    minf = tf(tf.xopt[1]) + pt(tf.xopt[1])
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
            Fminbox(LBFGS(
                linesearch = Optim.LineSearches.BackTracking(order=2)
            )),
            Optim.Options(x_reltol=X_RELTOL, f_reltol=F_RELTOL, time_limit=NEWTON_SOLVE_TIME_LIMIT)
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