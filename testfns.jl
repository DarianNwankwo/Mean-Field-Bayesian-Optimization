import Base:+, *, -
# https://www.sfu.ca/~ssurjano/optimization.html
# https://en.wikipedia.org/wiki/Test_functions_for_optimization

struct TestFunction{F, G, N}
    dim::Int
    bounds::Matrix{Float64}
    xopt::NTuple{N, Vector{Float64}}
    f::F
    ∇f!::G
end


(testfn::TestFunction)(x::AbstractVector{T}) where T = testfn.f(x)::Float64
gradient(testfn::TestFunction) = testfn.∇f

# Apply the function or its gradient to each column of the matrix
function (testfn::TestFunction)(X::Matrix{T}; grad=false) where T <: Real
    N = size(X, 2)
    y = zeros(T, N)

    @views begin
        for i in 1:N
            y[i] = testfn(X[:, i])
        end
    end

    return y
end


function get_collapsed_bounds(t1::TestFunction, t2::TestFunction)
    closest_to_origin(x::Vector{T}) where T = x[findmin(abs, x)[2]]

    bounds = zeros(T, t1.dim, 2)
    union_lowerbounds = [t1.bounds[:, 1] t2.bounds[:, 1]]
    union_upperbounds = [t1.bounds[:, 2] t2.bounds[:, 2]]

    for i = 1:t1.dim
        bounds[i, 1] = closest_to_origin(union_lowerbounds[i, :])
        bounds[i, 2] = closest_to_origin(union_upperbounds[i, :])
    end

    return bounds
end


function +(t1::TestFunction, t2::TestFunction)
    @assert t1.dim == t2.dim "dim(t1) must equal dim(t2)"
    return TestFunction(
        t1.dim,
        get_collapsed_bounds(t1, t2),
        (zeros(t1.dim),),
        (x) -> t1.f(x) + t2.f(x),
        (x) -> t1.∇f(x) + t2.∇f(x)
    )
end

function *(t1::TestFunction, t2::TestFunction)
    @assert t1.dim == t2.dim "dim(t1) must equal dim(t2)"
    return TestFunction(
        t1.dim,
        get_collapsed_bounds(t1, t2),
        (zeros(t1.dim),),
        (x) -> t1.f(x) * t2.f(x),
        (x) -> t1.f(x) * t2.∇f(x) + t1.∇f(x) * t2.f(x)
    )
end

function -(t1::TestFunction)
    return TestFunction(
        t1.dim,
        t1.bounds,
        t1.xopt,
        (x) -> -1. * t1.f(x),
        (x) -> -1. * t1.∇f(x)
    )
end


function scalar_scale(testfn::TestFunction, s::T) where T <: Number
    function f(x::Vector{T})
        return testfn.f(x / s)
    end
    function ∇f(x::Vector{T})
        return testfn.∇f(x / s) / s
    end
    return TestFunction(testfn.dim, testfn.bounds * s, testfn.xopt .* s, f, ∇f)
end


function vshift(testfn::TestFunction, s::T) where T <: Number
    function f(x::Vector{T})
        return testfn.f(x) + s
    end
    function ∇f(x::Vector{T})
        return testfn.∇f(x)
    end
    return TestFunction(testfn.dim, testfn.bounds, testfn.xopt, f, ∇f)
end

function hshift(testfn::TestFunction, s::Vector{T}) where T <: Number
    function f(x)
        return testfn.f(x + s)
    end
    function ∇f(x)
        return testfn.∇f(x)
    end
    return TestFunction(testfn.dim, testfn.bounds, testfn.xopt + s , f, ∇f)
end

function get_bounds(t::TestFunction)
    @views begin
        return (t.bounds[:, 1], t.bounds[:, 2])
    end
end

function tplot(f::TestFunction)
    if f.dim == 1
        xx = range(f.bounds[1,1], f.bounds[1,2], length=250)
        p = plot(xx, (x) -> f([x]), label="f(x)")
        scatter!(p, [xy[1] for xy in f.xopt], [f(xy) for xy in f.xopt], label="xopt")
        return p
    elseif f.dim == 2
        xx = range(f.bounds[1,1], f.bounds[1,2], length=100)
        yy = range(f.bounds[2,1], f.bounds[2,2], length=100)
        plot(xx, yy, (x,y) -> f([x,y]), st=:contour)
        scatter!([xy[1] for xy in f.xopt], [xy[2] for xy in f.xopt], label="xopt")
        # scatter!([f.xopt[1]], [f.xopt[2]], label="xopt")
    else
        error("Can only plot 1- or 2-dimensional TestFunctions")
    end
end

function TestLevy(d)
    f(x) = begin
        w = 1 .+ (x .- 1) ./ 4
        term1 = sin(π * w[1])^2
        sum_terms = sum((w[1:end-1] .- 1).^2 .* (1 .+ 10 .* sin.(π .* w[1:end-1] .+ 1).^2))
        term3 = (w[end] - 1)^2 * (1 + sin(2 * π * w[end])^2)
        term1 + sum_terms + term3
    end

    ∇f!(G, x) = begin
        G[:] = ForwardDiff.gradient(f, x)
        return G
    end

    bounds = zeros(d, 2)
    bounds[:, 1] .= -10.
    bounds[:, 2] .= 10.
    xopt = Tuple([ones(Float64, d)])  # Tuple containing the optimal vector

    return TestFunction(d, bounds, xopt, f, ∇f!)
end

function TestBraninHoo(; a=1, b=5.1/(4π^2), c=5/π, r=6, s=10, t=1/(8π))
    function f(xy)
        x = xy[1]
        y = xy[2]
        a*(y-b*x^2+c*x-r)^2 + s*(1-t)*cos(x) + s
    end
    function ∇f!(G, xy)
        x = xy[1]
        y = xy[2]
        dx = 2*a*(y-b*x^2+c*x-r)*(-b*2*x+c) - s*(1-t)*sin(x)
        dy = 2*a*(y-b*x^2+c*x-r)
        G[1] = dx
        G[2] = dy
        return G
    end
    bounds = [-5.0 10.0 ; 0.0 15.0]
    xopt = Tuple([[-π, 12.275], [π, 2.275], [9.42478, 2.475]])
    return TestFunction(2, bounds, xopt, f, ∇f!)
end


function TestRosenbrock()
    f(xy) = (1-xy[1])^2 + 100*(xy[2]-xy[1]^2)^2
    ∇f!(G, xy) = begin
        G[1] = -2*(1-xy[1]) - 400*xy[1]*(xy[2]-xy[1]^2)
        G[2] = 200*(xy[2]-xy[1]^2)
        return G
    end
    return TestFunction(2, [-2.0 2.0 ; -1.0 3.0 ], Tuple([ones(2)]), f, ∇f!)
end


function TestRastrigin(n)
    # Refactored function evaluation without allocations.
    f(x) = begin
        s = 0.0
        for i in 1:length(x)
            s += x[i]^2 - 10 * cos(2π * x[i])
        end
        return 10*n + s
    end

    # Refactored in-place gradient evaluation to avoid allocations.
    ∇f!(G, x) = begin
        for i in eachindex(x)
            G[i] = 2*x[i] + 20π*sin(2π * x[i])
        end
        return G
    end

    # Allocate bounds and set each column elementwise.
    bounds = zeros(n, 2)
    for i in 1:n
        bounds[i, 1] = -5.12
        bounds[i, 2] = 5.12
    end

    xopt = Tuple([zeros(n)])
    return TestFunction(n, bounds, xopt, f, ∇f!)
end


function TestAckley(d; a=20.0, b=0.2, c=2π)
    
    function f(x)
        cx = 0.0
        @inbounds @simd for i in eachindex(x)
            cx += cos(c * x[i])
        end
        nx = sqrt(sum(@inbounds x[i]^2 for i in eachindex(x)))
        return -a * exp(-b / sqrt(d) * nx) - exp(cx / d) + a + exp(1)
    end
    
    function ∇f!(grad, x)
        nx = norm(x)
        if nx == 0.0
            grad[:] = 0.
            return grad
        end
    
        # Compute the sum of cos(c*x) without allocating an array.
        cx = 0.0
        for xi in x
            cx += cos(c * xi)
        end
    
        # Precompute scalar factors; note we reuse nx (computed once) here.
        factor1 = (a * b) / sqrt(d) * exp(-b / sqrt(d) * nx)
        factor2 = exp(cx / d) / d
    
        # Compute the gradient in one loop, combining the normalization and sine evaluation.
        for i in eachindex(x)
            # Instead of forming dnx and dcx as separate arrays, we compute
            # grad[i] directly as:
            #   factor1 * (x[i] / nx) - factor2 * (-c*sin(c*x[i]))
            # which simplifies to:
            grad[i] = factor1 * (x[i] / nx) + factor2 * c * sin(c * x[i])
        end
    
        return grad
    end

    bounds = zeros(d,2)
    bounds[:,1] .= -32.768
    bounds[:,2] .=  32.768
    xopt = Tuple([zeros(d)])

    return TestFunction(d, bounds, xopt, f, ∇f!)
end


function TestSixHump()

    function f(xy)
        x = xy[1]
        y = xy[2]
        xterm = (4.0-2.1*x^2+x^4/3)*x^2
        yterm = (-4.0+4.0*y^2)*y^2
        xterm + x*y + yterm
    end

    function ∇f!(G, xy)
        x = xy[1]
        y = xy[2]
        dxterm = (-4.2*x+4.0*x^3/3)*x^2 + (4.0-2.1*x^2+x^4/3)*2.0*x
        dyterm = (8.0*y)*y^2 + (-4.0+4.0*y^2)*2.0*y
        G[1] = dxterm + y
        G[2] = dyterm + x
        return G
    end

    # There's a symmetric optimum
    xopt = Tuple([[0.089842, -0.712656], [-0.089842, 0.712656]])

    return TestFunction(2, [-3.0 3.0 ; -2.0 2.0], xopt, f, ∇f!)
end


function TestGramacyLee()
    f(x) = sin(10π*x[1])/(2*x[1]) + (x[1]-1.0)^4
    ∇f!(G, x) = begin
        G[1] = 5π*cos(10π*x[1])/x[1] - sin(10π*x[1])/(2*x[1]^2) + 4*(x[1]-1.0)^3
        return G
    end 
    bounds = zeros(1, 2)
    bounds[1,1] = 0.5
    bounds[1,2] = 2.5
    xopt = Tuple([[0.548563]])
    return TestFunction(1, bounds, xopt, f, ∇f!)
end


function TestGoldsteinPrice()
    function f(xy)
        x1 = xy[1]
        x2 = xy[2]
        t1 = x1 + x2 + 1
        t2 = 2 * x1 - 3 * x2
        t3 = x1^2
        t4 = x2^2
        
        term1 = 1 + t1^2 * (19 - 14 * x1 + 3 * t3 - 14 * x2 + 6 * x1 * x2 + 3 * t4)
        term2 = 30 + t2^2 * (18 - 32 * x1 + 12 * t3 + 48 * x2 - 36 * x1 * x2 + 27 * t4)
        
        return term1 * term2
    end
    
    function ∇f!(G, xy)
        x1 = xy[1]
        x2 = xy[2]
        t1 = x1 + x2 + 1
        t2 = 2 * x1 - 3 * x2
        t3 = x1^2
        t4 = x2^2
        
        common1 = 2 * t1 * (3 * t3 + 6 * x1 * x2 - 14 * x1 + 3 * t4 - 14 * x2 + 19)
        common2 = t2^2 * (12 * t3 - 36 * x1 * x2 + 18 - 32 * x1 + 27 * t4 + 48 * x2)
        
        df1 = common1 + common2 * (2 * t3 - 36 * x1 * x2 - 32 * x2 + 48 * t4 + 18 - 32 * x1)
        df2 = common1 + common2 * (48 * x1 - 36 * x1 * x2 - 32 * x2 + 27 * t4 + 18 - 32 * x1)
        
        G[1] = df1
        G[2] = df2
        return G
    end

    bounds = zeros(2, 2)
    bounds[:,1] .= -2.0
    bounds[:,2] .=  2.0

    xopt = Tuple([[0.0, -1.0]])

    return TestFunction(2, bounds, xopt, f, ∇f!)
end


function TestBeale()
    function f(xy)
        x1 = xy[1]
        x2 = xy[2]
        t1 = 1.5 - x1 + x1 * x2
        t2 = 2.25 - x1 + x1 * x2^2
        t3 = 2.625 - x1 + x1 * x2^3
        return t1^2 + t2^2 + t3^2
    end
    
    function ∇f!(G, xy)
        x1 = xy[1]
        x2 = xy[2]
        t1 = 1.5 - x1 + x1 * x2
        t2 = 2.25 - x1 + x1 * x2^2
        t3 = 2.625 - x1 + x1 * x2^3
        
        df1 = 2 * (t1 * (x2 - 1) + t2 * (x2^2 - 1) + t3 * (x2^3 - 1))
        df2 = 2 * (t1 * x1 + 2 * t2 * x1 * x2 + 3 * t3 * x1 * x2^2)
        G[1] = df1
        G[2] = df2

        return G
    end

    bounds = zeros(2, 2)
    bounds[:,1] .= -4.5
    bounds[:,2] .=  4.5

    xopt = Tuple([[3.0, 0.5]])

    return TestFunction(2, bounds, xopt, f, ∇f!)
end


function TestEasom()
    f(x) = -cos(x[1])*cos(x[2])*exp(-((x[1]-π)^2 + (x[2]-π)^2))

    function ∇f!(G, x)
        c = cos(x[1]) * cos(x[2])
        e = exp(-((x[1] - π)^2 + (x[2] - π)^2))
        term = 2 * (x[1] - π) * cos(x[2]) + 2 * (x[2] - π) * cos(x[1])
        
        df1 = c * e * term - sin(x[1]) * cos(x[2]) * e
        df2 = c * e * term - sin(x[2]) * cos(x[1]) * e
        G[1] = df1
        G[2] = df2

        return G
    end

    bounds = zeros(2, 2)
    bounds[:, 1] .= -100.0
    bounds[:, 2] .= 100.0
    
    xopt = Tuple([[float(π), float(π)]])

    return TestFunction(2, bounds, xopt, f, ∇f!)
end


function TestStyblinskiTang(d)
    # Allocation-free function evaluation
    f(x) = begin
        s = 0.0
        @inbounds for i in eachindex(x)
            s += x[i]^4 - 16*x[i]^2 + 5*x[i]
        end
        return 0.5 * s
    end

    # Allocation-free in-place gradient evaluation
    function ∇f!(G, x)
        @inbounds for i in eachindex(x)
            G[i] = 2*x[i]^3 - 16*x[i] + 2.5
        end
        return G
    end

    # Set up bounds without extra allocations
    bounds = zeros(d, 2)
    for i in 1:d
        bounds[i, 1] = -5.0
        bounds[i, 2] = 5.0
    end

    xopt = Tuple([repeat([-2.903534], d)])
    return TestFunction(d, bounds, xopt, f, ∇f!)
end


function TestBukinN6()
    function f(x)
        x1 = x[1]
        x2 = x[2]
        t1 = abs(x2 - 0.01 * x1^2)
        
        term1 = 100 * sqrt(t1) + 0.01 * abs(x1 + 10)
        return term1
    end
    
    function ∇f!(G, x)
        x1 = x[1]
        x2 = x[2]
        t1 = abs(x2 - 0.01 * x1^2)
        t2 = sqrt(t1)
        
        df1 = 0.01 * x1 / t2 + 0.01
        df2 = 50 * (x2 - 0.01 * x1^2) / t2
        G[1] = df1
        G[2] = df2
        return G
    end

    bounds = zeros(2, 2)
    bounds[:,1] .= -15.0
    bounds[:,2] .=  3.0
    xopt = Tuple([[-10.0, 1.0]])
    return TestFunction(2, bounds, xopt, f, ∇f!)
end


function TestCrossInTray()
    f(x) = -0.0001 * (abs(sin(x[1]) * sin(x[2]) * exp(abs(100 - sqrt(x[1]^2 + x[2]^2) / π))) + 1)^0.1

    function ∇f!(G, x)
        # Extract variables
        x1 = x[1]
        x2 = x[2]
        s1 = sin(x1)
        s2 = sin(x2)
        A = s1 * s2
        dA_dx1 = cos(x1) * s2
        dA_dx2 = s1 * cos(x2)
        
        # Compute the norm and protect against division by zero
        R = sqrt(x1^2 + x2^2)
        if R < eps(x1)
            t1_x = 0.0
            t1_y = 0.0
        else
            Z = 100 - R/π
            sign_Z = Z > 0 ? 1.0 : (Z < 0 ? -1.0 : 0.0)
            t1_x = A * sign_Z * (x1/(π * R))
            t1_y = A * sign_Z * (x2/(π * R))
        end
        
        # Compute B, v, and u
        Z = 100 - R/π  # recompute to ensure definition
        sign_Z = Z > 0 ? 1.0 : (Z < 0 ? -1.0 : 0.0)
        B = exp(abs(Z))
        v = A * B
        u = abs(v) + 1.0
        
        # sign(v) is the sign of A since B > 0
        sign_v = A > 0 ? 1.0 : (A < 0 ? -1.0 : 0.0)
        
        # Compute the common factor from the chain rule
        factor = -0.00001 * u^(-0.9) * B * sign_v
        
        # Compute the gradient components
        G[1] = factor * (dA_dx1 - (R < eps(x1) ? 0.0 : t1_x))
        G[2] = factor * (dA_dx2 - (R < eps(x1) ? 0.0 : t1_y))
        return G
    end

    bounds = zeros(2, 2)
    bounds[:,1] .= -10.0
    bounds[:,2] .=  10.0
    xopt = Tuple([repeat([1.34941], 2)])
    return TestFunction(2, bounds, xopt, f, ∇f!)
end


function TestEggHolder()
    f(x) = -(x[2] + 47) * sin(sqrt(abs(x[2] + x[1] / 2 + 47))) - x[1] * sin(sqrt(abs(x[1] - (x[2] + 47))))
    function ∇f!(G, x)
        x1 = x[1]
        x2 = x[2]
        
        # Define intermediate expressions
        A = x2 + 0.5*x1 + 47.0         # A = x[2] + x[1]/2 + 47
        B = x1 - (x2 + 47.0)           # B = x[1] - (x[2] + 47)
        
        absA = abs(A)
        absB = abs(B)
        
        # Compute square roots of absolute values
        sqrtA = sqrt(absA)
        sqrtB = sqrt(absB)
        
        # Compute factors for the chain rule; protect against division by zero
        factorA = absA == 0.0 ? 0.0 : cos(sqrtA) / (2 * sqrtA) * sign(A)
        factorB = absB == 0.0 ? 0.0 : cos(sqrtB) / (2 * sqrtB) * sign(B)
        
        # Compute derivative with respect to x1
        dfdx1_f1 = - (x2 + 47.0) * factorA * 0.5
        dfdx1_f2 = - sin(sqrtB) - x1 * factorB
        G[1] = dfdx1_f1 + dfdx1_f2
        
        # Compute derivative with respect to x2
        dfdx2_f1 = - sin(sqrtA) - (x2 + 47.0) * factorA
        dfdx2_f2 = x1 * factorB
        G[2] = dfdx2_f1 + dfdx2_f2
        
        return G
    end
    bounds = zeros(2, 2)
    bounds[:,1] .= -512.0
    bounds[:,2] .=  512.0
    xopt = Tuple([[512, 404.2319]])
    return TestFunction(2, bounds, xopt, f, ∇f!)
end


function TestHolderTable()
    f(x) = -abs(sin(x[1]) * cos(x[2]) * exp(abs(1 - sqrt(x[1]^2 + x[2]^2) / π)))
    
    function ∇f!(G, x)
        # Extract variables
        x1 = x[1]
        x2 = x[2]
        R = sqrt(x1^2 + x2^2)
        
        # Compute A, T, U, and B
        A = sin(x1) * cos(x2)
        T = 1 - R/π
        U = abs(T)
        B = exp(U)
        
        # Compute z and its sign
        z = A * B
        sign_z = z > 0 ? 1.0 : (z < 0 ? -1.0 : 0.0)
        
        # Compute derivative of A
        dA_dx1 = cos(x1) * cos(x2)
        dA_dx2 = -sin(x1) * sin(x2)
        
        # Compute derivative of z = A * B
        if R < eps(x1)
            # When R is nearly zero, avoid division by zero
            dz_dx1 = B * dA_dx1
            dz_dx2 = B * dA_dx2
        else
            sign_T = T > 0 ? 1.0 : (T < 0 ? -1.0 : 0.0)
            common = (A * sign_T) / (π * R)
            dz_dx1 = B * (dA_dx1 - common * x1)
            dz_dx2 = B * (dA_dx2 - common * x2)
        end
        
        # Gradient of f(x) = -|z| is -sign(z) * dz/dx
        G[1] = -sign_z * dz_dx1
        G[2] = -sign_z * dz_dx2
        
        return G
    end
    bounds = zeros(2, 2)
    bounds[:,1] .= -10.0
    bounds[:,2] .=  10.0
    xopt = Tuple([[8.05502, 9.66459]])
    return TestFunction(2, bounds, xopt, f, ∇f!)
end


function TestSchwefel(d)
    f(x) = 418.9829 * d - sum(x .* sin.(sqrt.(abs.(x))))
    
    function ∇f!(G, x)
        @inbounds for i in eachindex(x)
            xi = x[i]
            absxi = abs(xi)
            if absxi < eps(xi)
                dfdx = 0.0
            else
                s = sqrt(absxi)
                dfdx = sin(s) + (xi * cos(s) * sign(xi)) / (2 * s)
            end
            G[i] = - dfdx
        end
        return G
    end
    
    bounds = zeros(d, 2)
    bounds[:,1] .= -500.0
    bounds[:,2] .=  500.0
    xopt = Tuple([repeat([420.9687], d)])
    return TestFunction(d, bounds, xopt, f, ∇f!)
end


function TestLevyN13()
    f(x) = sin(3π * x[1])^2 + (x[1] - 1)^2 * (1 + sin(3π * x[2])^2) + (x[2] - 1)^2 * (1 + sin(2π * x[2])^2)

    function ∇f!(G, x)
        x1, x2 = x[1], x[2]
        s1, c1 = sin(3π * x1), cos(3π * x1)
        s2, c2 = sin(3π * x2), cos(3π * x2)
        s3, c3 = sin(2π * x2), cos(2π * x2)

        G[1] = 2 * s1 * c1 * 3π + 2 * (x1 - 1) * (1 + s2^2)
        G[2] = (x1 - 1)^2 * 2 * s2 * c2 * 3π +
               2 * (x2 - 1) * (1 + s3^2) +
               (x2 - 1)^2 * 2 * s3 * c3 * 2π
        return G
    end

    bounds = [-10.0 10.0; -10.0 10.0]
    xopt = Tuple([[1.0, 1.0]])
    return TestFunction(2, bounds, xopt, f, ∇f!)
end


function TestTrid(d)
    f(x) = sum((x .- 1).^2) - sum(x[2:end] .* x[1:end-1])
    
    function ∇f!(G, x)
        d = length(x)
        if d == 0
            return G
        end
        # Compute gradient for the first element
        G[1] = 2*(x[1] - 1) - (d > 1 ? x[2] : 0.0)
        # Compute gradient for middle elements, if any
        for i in 2:d-1
            G[i] = 2*(x[i] - 1) - (x[i-1] + x[i+1])
        end
        # Compute gradient for the last element, if d > 1
        if d > 1
            G[d] = 2*(x[d] - 1) - x[d-1]
        end
        return G
    end
    
    bounds = zeros(d, 2)
    bounds[:,1] .= -d^2
    bounds[:,2] .=  d^2
    xopt = Tuple([[i * (d + 1. - i) for i in 1:d]])
    return TestFunction(d, bounds, xopt, f, ∇f!)
end


function TestMccormick()
    f(x) = sin(x[1] + x[2]) + (x[1] - x[2])^2 - 1.5 * x[1] + 2.5 * x[2] + 1
    function ∇f!(G, x)
        x1 = x[1]
        x2 = x[2]
        G[1] = cos(x1 + x2) + 2*(x1 - x2) - 1.5
        G[2] = cos(x1 + x2) - 2*(x1 - x2) + 2.5
        return G
    end
    bounds = zeros(2, 2)
    bounds[:,1] .= -1.5
    bounds[:,2] .= 4.0
    xopt = Tuple([[-0.54719, -1.54719]])
    return TestFunction(2, bounds, xopt, f, ∇f!)
end


function TestHartmann3D()
    α = [1.0, 1.2, 3.0, 3.2]
    A = [
        3.0 10 30;
        0.1 10 35;
        3.0 10 30;
        0.1 10 35
    ]
    P = 1e-4 * [
        3689 1170 2673;
        4699 4387 7470;
        1091 8732 5547;
        381 5743 8828
    ]
    
    function f(x)
        f_val = 0.0
        for i in 1:4
            t = 0.0
            for j in 1:3
                t += A[i,j] * (x[j] - P[i,j])^2
            end
            f_val += α[i] * exp(-t)
        end
        return -f_val
    end
    
    function ∇f!(G, x)
        fill!(G, 0.0)
        for i in 1:4
            t = 0.0
            for j in 1:3
                t += A[i,j] * (x[j] - P[i,j])^2
            end
            exp_neg_t = exp(-t)
            for k in 1:3
                G[k] += 2 * α[i] * A[i,k] * (x[k] - P[i,k]) * exp_neg_t
            end
        end
        return G
    end
    
    bounds = zeros(3, 2)
    bounds[:,1] .= 0.0
    bounds[:,2] .= 1.0
    xopt = Tuple([[0.114614, 0.555649, 0.852547]])
    return TestFunction(3, bounds, xopt, f, ∇f!)
end


function TestHartmann4D()
    α = [1.0, 1.2, 3.0, 3.2]
    # A and P are 4×6, but only columns 1..4 are used for x[1..4]
    A = [
        10    3    17   3.5   1.7   8;
        0.05  10   17   0.1   8     14;
        3     3.5  1.7  10    17    8;
        17    8    0.05 10    0.1   14
    ]
    P = 1e-4 * [
        1312 1696 5569 124  8283 5886;
        2329 4135 8307 3736 1004 9991;
        2348 1451 3522 2883 3047 6650;
        4047 8828 8732 5743 1091 381
    ]

    function f(x)
        # x is 4-dimensional; ignore columns 5..6 of A and P
        sum_val = 0.0
        for i in 1:4
            t = 0.0
            for j in 1:4
                t += A[i, j] * (x[j] - P[i, j])^2
            end
            sum_val += α[i] * exp(-t)
        end
        return (1 / 0.839) * (1.1 - sum_val)
    end

    function ∇f!(G, x)
        # Zero out G to accumulate in-place
        fill!(G, 0.0)

        # For each row i in A, P
        for i in 1:4
            # Compute tᵢ = Σⱼ A[i,j]*(xⱼ - P[i,j])², j=1..4
            t = 0.0
            for j in 1:4
                t += A[i, j] * (x[j] - P[i, j])^2
            end

            # Precompute αᵢ * e^(−tᵢ)
            fac = α[i] * exp(-t)

            # Accumulate partial derivatives in each dimension k
            for k in 1:4
                # ∂/∂xₖ of exp(-tᵢ) = exp(-tᵢ)*[ -∂tᵢ/∂xₖ ], but the minus sign is
                # canceled by the negative sign in the function definition,
                # so it becomes a plus in the final expression.
                # Here, ∂tᵢ/∂xₖ = 2*A[i,k]*(xₖ - P[i,k])
                G[k] += 2 * A[i, k] * (x[k] - P[i, k]) * fac
            end
        end

        # Multiply by the outer constant factor (1 / 0.839)
        @inbounds for k in 1:4
            G[k] *= (1 / 0.839)
        end

        return G
    end

    bounds = zeros(4, 2)
    bounds[:,1] .= 0.0
    bounds[:,2] .= 1.0
    # Known reference optimum in 4D (still uses only the first 4 coords)
    xopt = Tuple([[0.20169, 0.150011, 0.476874, 0.275332]])

    return TestFunction(4, bounds, xopt, f, ∇f!)
end

function TestHartmann6D()
    α = [1.0, 1.2, 3.0, 3.2]
    A = [
        10    3    17   3.5   1.7   8;
        0.05  10   17   0.1   8     14;
        3     3.5  1.7  10    17    8;
        17    8    0.05 10    0.1   14
    ]
    P = 1e-4 * [
        1312 1696 5569 124  8283 5886;
        2329 4135 8307 3736 1004 9991;
        2348 1451 3522 2883 3047 6650;
        4047 8828 8732 5743 1091 381
    ]

    # The function f(x) = -∑ᵢ αᵢ exp(-tᵢ), where tᵢ = ∑ⱼ Aᵢⱼ (xⱼ - Pᵢⱼ)²
    function f(x)
        total = 0.0
        for i in 1:4
            t = 0.0
            for j in 1:6
                t += A[i,j] * (x[j] - P[i,j])^2
            end
            total += α[i] * exp(-t)
        end
        return -total
    end

    # In-place gradient of f, ∇f!(G, x)
    function ∇f!(G, x)
        fill!(G, 0.0)  # zero out the gradient array first
        for i in 1:4
            # Compute tᵢ = ∑ⱼ Aᵢⱼ (xⱼ - Pᵢⱼ)²
            t = 0.0
            for j in 1:6
                t += A[i,j] * (x[j] - P[i,j])^2
            end
            # Each term contributes αᵢ exp(-tᵢ) to the sum,
            # so the partial derivative wrt xₖ is:
            #   d/dxₖ [ - αᵢ exp(-tᵢ) ] =  2 αᵢ Aᵢₖ (xₖ - Pᵢₖ) exp(-tᵢ)
            #   (the sign flips twice: once for the negative outside,
            #    and once for the derivative of exp(-tᵢ).)
            factor = α[i] * exp(-t)
            for k in 1:6
                G[k] += 2 * A[i,k] * (x[k] - P[i,k]) * factor
            end
        end
        return G
    end

    # Bounds, optimum, and TestFunction constructor
    bounds = zeros(6, 2)
    bounds[:,1] .= 0.0
    bounds[:,2] .= 1.0
    xopt = Tuple([[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]])
    return TestFunction(6, bounds, xopt, f, ∇f!)
end


function TestConstant(n=0.; lbs::Vector{<:T}, ubs::Vector{<:T}) where T <: Real
    f(x) = n
    ∇f!(G, x) = begin
        fill!(G, 0.)
        return G
    end
    xopt = Tuple([zeros(length(lbs))])
    bounds = hcat(lbs, ubs)
    return TestFunction(length(lbs), bounds, xopt, f, ∇f!)
end


function TestQuadratic1D(a=1, b=0, c=0; lb=-1.0, ub=1.0)
    f(x) = a*first(x)^2 + b*first(x) + c
    ∇f!(G, x) = begin
        G[1] = 2*a*first(x) + b
    end
    bounds = [lb ub]
    xopt = Tuple([zeros(1)])
    return TestFunction(1, bounds, xopt, f, ∇f!)
end

# Create a test function named TestLinearCosine1D that takes a paramater for the frequency of the cosine
# and a parameter for the amplitude of the cosine. The function should be a linear function of x plus a cosine
# function of the form a*cos(b*x). The function should have a single optimum at x=0.

function TestLinearCosine1D(a=1, b=1; lb=-1.0, ub=1.0)
    f(x) = a*first(x) * cos(b*first(x))
    ∇f(G, x) = begin
        G[1] = a*cos(b*first(x)) - a*b*first(x)*sin(b*first(x))
        return G
    end
    bounds = [lb ub]
    xopt = Tuple([zeros(1)])
    return TestFunction(1, bounds, xopt, f, ∇f!)
end



function TestShekel()
    C = [
         4. 1. 8. 6. 3. 2. 5. 8. 6. 7.;
         4. 1. 8. 6. 7. 9. 3. 1. 2. 3.;
         4. 1. 8. 6. 3. 2. 5. 8. 6. 7.;
         4. 1. 8. 6. 7. 9. 3. 1. 2. 3.
    ]
    β = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    m = 10

    function f(x)
        f = 0.0
        for i in 1:m
            t = 0.0
            for j in 1:4
                t += (x[j] - C[j,i])^2
            end
            f += 1 / (t + β[i])
        end
        return -f
    end

    function ∇f!(G, x)
        for i in 1:m
            t = 0.0
            for j in 1:4
                t += (x[j] - C[j,i])^2
            end
            for j in 1:4
                G[j] += 2 * (x[j] - C[j,i]) / (t + β[i])^2
            end
        end
        G[:] = -G[:]
        return G
    end

    bounds = [zeros(4) 10ones(4)]
    xopt = Tuple([[4.0, 4.0, 4.0, 4.0]])
    return TestFunction(4, bounds, xopt, f, ∇f!)
end


function TestDropWave()
    f(x) = begin
        sum_x = x[1]*x[1] + x[2]*x[2]
        -(1 + cos(12*sqrt(sum_x))) / (0.5*sum_x + 2)
    end
    
    function ∇f!(G, x)
        sum_x = x[1]*x[1] + x[2]*x[2]
        t = 12 * sqrt(sum_x)
        for i in 1:2
            G[i] = 12 * x[i] * sin(t) / sqrt(sum_x) / (0.5*sum_x + 2) - (1 + cos(t)) * (x[i] / (0.5*sum_x + 2)^2)
        end
        G[:] = -G[:]
        return G
    end

    bounds = zeros(2, 2)
    bounds[:,1] .= -5.12
    bounds[:,2] .=  5.12
    xopt = Tuple([[0.0, 0.0]])
    return TestFunction(2, bounds, xopt, f, ∇f!)
end


function TestBohachevsky()
    f(x) = x[1]^2 + 2*x[2]^2 - 0.3*cos(3π*x[1]) - 0.4*cos(4π*x[2]) + 0.7
    
    function ∇f!(G, x)
        G[1] = 2*x[1] + 0.9*π*sin(3π*x[1])
        G[2] = 4*x[2] + 1.6*π*sin(4π*x[2])
        return G
    end

    bounds = zeros(2, 2)
    bounds[:,1] .= -100.0
    bounds[:,2] .=  100.0
    xopt = Tuple([[0.0, 0.0]])
    return TestFunction(2, bounds, xopt, f, ∇f!)
end


function TestGriewank(d)
    function f(x)
        sum = 0.0
        product = 1.0
        for i in 1:length(x)
            sum += x[i]^2
            product *= cos(x[i]/sqrt(i))
        end
        return 1 + sum/4000 - product
    end
    
    function ∇f!(G, x)
        for i in 1:length(x)
            sin_term = sin(x[i]/sqrt(i))
            cos_term = prod([cos(x[j]/sqrt(j)) for j=1:length(x) if j != i])
            G[i] = 2*x[i]/4000 + sin_term * cos_term
        end
        return G
    end

    bounds = zeros(d, 2)
    bounds[:, 1] .= -600.
    bounds[:, 2] .= 600.
    xopt = Tuple([zeros(d)])

    return TestFunction(d, bounds, xopt, f, ∇f!)
end


# Utility: normalize any TestFunction to the unit hypercube [0,1]^d
function normalize_testfn(tf::TestFunction)
    lower, upper = get_bounds(tf)
    d = tf.dim

    # Compute Δ = upper - lower without broadcast
    Δ = Vector{Float64}(undef, d)
    for i in 1:d
        Δ[i] = upper[i] - lower[i]
    end

    # Preallocate a buffer for mapped-back points
    x_orig = Vector{Float64}(undef, d)

    # Unit-space function
    function f_unit(x::AbstractVector{<:Real})
        for i in 1:d
            x_orig[i] = lower[i] + Δ[i] * x[i]
        end
        return tf.f(x_orig)
    end

    # Unit-space in-place gradient
    function ∇f_unit!(G::AbstractVector{<:Real}, x::AbstractVector{<:Real})
        for i in 1:d
            x_orig[i] = lower[i] + Δ[i] * x[i]
        end
        tf.∇f!(G, x_orig)
        for i in 1:d
            G[i] *= Δ[i]
        end
        return G
    end

    # New [0,1]^d bounds without broadcast
    bounds_unit = Matrix{Float64}(undef, d, 2)
    for i in 1:d
        bounds_unit[i, 1] = 0.0
        bounds_unit[i, 2] = 1.0
    end

    # Map original xopt to unit coords without broadcast
    xopt_unit_vec = Vector{Float64}(undef, d)
    for i in 1:d
        xopt_unit_vec[i] = (tf.xopt[1][i] - lower[i]) / Δ[i]
    end
    xopt_unit = (xopt_unit_vec,)

    return TestFunction(d, bounds_unit, xopt_unit, f_unit, ∇f_unit!)
end


function TestForrester1D()
    # The black‐box objective
    f(x) = (6x[1] - 2)^2 * sin(12x[1] - 4)

    # In‐place gradient ∇f!
    function ∇f!(G, x)
        x1 = x[1]
        y  = 6x1 - 2            # helper
        u  = y^2
        v  = sin(12x1 - 4)
        # du/dx = 12*y, dv/dx = 12*cos(12x1 - 4)
        G[1] = 12y * v + u * 12 * cos(12x1 - 4)
        return nothing
    end

    # Domain is [0,1]
    bounds = zeros(1, 2)
    bounds[:, 1] .= 0.0
    bounds[:, 2] .= 1.0

    # Approximate global minimizer
    xopt = ([0.7572],)

    return TestFunction(1, bounds, xopt, f, ∇f!)
end