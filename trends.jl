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
    
    return TestFunction(
        testfn.dim,
        testfn.bounds,
        testfn.xopt,
        new_f,
        new_∇f
    )
end