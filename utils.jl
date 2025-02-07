"""
Generate low discrepancy Sobol sequence of uniform random variables
"""
function gen_uniform(samples; dim=1)
    sobol = SobolSeq(zeros(dim), ones(dim))
    S = zeros(dim, samples)
    
    for j in 1:samples
        S[:,j] = next!(sobol)
    end
    
    S
end

function is_odd(num)
    return mod(num,2) == 1
end

"""
Transforms an even-sized multivariate uniform distribution to be normally
distributed with mean 0 and variance 1.
"""
function box_muller_transform(S)
    dim, samples = size(S)
    N = zeros(dim, samples)
    
    for j in 1:samples
        y = zeros(dim)
        x = S[:,j]
        
        for i in 1:dim
            if isodd(i)
                y[i] = sqrt(-2log10(x[i]))*cos(2π*x[i+1])
            else
                y[i] = sqrt(-2log10(x[i-1]))*sin(2π*x[i])
            end
        end
        
        N[:,j] = y
    end
    
    N
end

dense_1D_discretization(;lb, ub, stepsize) = lb:stepsize:ub

"""
Produces a sequence of standard normally distributed values in 1D
"""
function uniform1d_to_normal(samples)
    uniform2d = gen_uniform(samples, dim=2)
    normal2d = box_muller_transform(uniform2d)
    marginal_normal = normal2d[1,:]
    
    marginal_normal
end

"""
Generate a low discrepancy multivariate normally distributed sequence
for monte carlo simulation of rollout acquisition functions with a max
horizon of h. The resulting tensor will be of size Mx(D+1)xH, where M
is the number of Monte Carlo iterations, D is the dimension of the
input, and H is our max horizon.
"""
function gen_low_discrepancy_sequence(samples, dim, horizon)
    # We need function and gradient samples, so dim here corresponds to the input space.
    # The +1 here corresponds to the function observation.
    offset = isodd(dim+1) ? 1 : 0
    S = gen_uniform(samples*horizon, dim=dim+1+offset)
    N = box_muller_transform(S)
    N = reshape(N, samples, dim+1+offset, horizon)
    
    return N[:,1:end-offset,:]
end

function randsample(N, d, lbs, ubs)
    X = zeros(d, N)
    for j = 1:N
        for i = 1:d
            X[i,j] = rand(Uniform(lbs[i], ubs[i]))
        end
    end
    return X
end


function stdize(series; a=0, b=1)
    smax, smin = maximum(series), minimum(series)
    return [a + (s - smin) / (smax - smin) * (b - a) for s in series]
end


centered_fd(f, u, du, h) = (f(u+h*du)-f(u-h*du)) / (2h)

function gap(initial_best::T, observed_best::T, actual_best::T) where T <: Real
    return (initial_best - observed_best) / (initial_best - actual_best)
end

function update_gaps!(gaps::Vector{T}, observations::Vector{T}, actual_best::T; start_index::Int = 1) where T <: Real
    @views begin
        finish_index = length(observations)
        initial_best = minimum(observations[1:start_index])

        for (j, end_index) in enumerate(start_index:finish_index)
            gaps[j] = gap(initial_best, minimum(observations[1:end_index]), actual_best)
        end
    end
    
    return nothing
end

simple_regret(actual_minimum::T, observation::T) where T <: Real = observation - actual_minimum

function generate_initial_guesses(N::Integer, lbs::Vector{T}, ubs::Vector{T},) where T <: Number
    ϵ = 1e-6
    seq = SobolSeq(lbs, ubs)
    initial_guesses = reduce(hcat, next!(seq) for i = 1:N)
    initial_guesses = hcat(initial_guesses, lbs .+ ϵ)
    initial_guesses = hcat(initial_guesses, ubs .- ϵ)

    return initial_guesses
end

function create_csv(filename::String, budget::Int)
    # Create the header based on the given budget
    header = ["trial"; string.(1:budget)...]

    # Create an empty DataFrame with the specified header
    df = DataFrame(-ones(1, length(header)), Symbol.(header))

    # Write the dataframe to a CSV file
    CSV.write(filename * ".csv", df)
end

function write_to_csv(filename::String, data::Vector{T}) where T <: Real
    CSV.write(
        filename * ".csv",
        Tables.table(data'),
        append=true
    )
end