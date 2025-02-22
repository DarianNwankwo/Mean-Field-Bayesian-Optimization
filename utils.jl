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



@doc raw"""
The experiments we want to conduct will inform us of the utility of the proposed hybrid model for doing Bayesian optimization when you know something about the mean field a priori. Our baseline model will be a standard zero mean Gaussian process. We will use a common set of synthetic test functions to evaluate our optimization algorithm. For each synthetic function, we want to consider the following perturbations:
1. Zero mean field trend
2. Constant mean field trend
3. Linear mean field trend
4. Nonlinear mean field trend
For each one these considerations, we also want to measure how standard acquisition functions absorb the information presented by the statistical model of choice.

For each of the perturbations specified, we want to construct a surrogate that has each of those trends encoded into their model. This, in some sense, is to measure model misspecification. So, for each test function, we’ll have a trend added to it and construct 4 models and perform the full Bayesian optimization loop (with EI, POI, and LCB) and collect the following metrics:
1. Simple regret
2. Gap
3. Model’s observation at each iteration
All of this metadata needs to be maintained as well. The structure might look something like the following:
1. <function_name>
    1. no_trend
        * zero_trend_surrogate
            * gap.csv
            * simple_regret.csv
            * observations.csv
            * minimum_observations.csv
            * trend.txt
        * constant_trend_surrogate
            * gap.csv
            * simple_regret.csv
            * observations.csv
            * minimum_observations.csv
            * trend.txt
        * linear_trend_surrogate
            * gap.csv
            * simple_regret.csv
            * observations.csv
            * minimum_observations.csv
            * trend.txt
        * nonlinear_trend_surrogate
            * gap.csv
            * simple_regret.csv
            * observations.csv
            * minimum_observations.csv
            * trend.txt
        * trend.txt (contains exact trend added to the test function)
    2. constant_trend
    3. linear_trend
    4. nonlinear_trend
    5. global_minimizer.txt
Where the first child directory under the function name denotes the test function that is being evaluated. The trend term is encoded in the directory’s name.
"""
function create_directory_structure(
    function_name,
    function_trends,
    surrogate_trends)
    # Parent directory of script being executed
    experiments_dir = dirname(abspath(PROGRAM_FILE))

    # Create the path for the sibling `data` directory
    data_dir = joinpath(experiments_dir, "data")

    # Create the path for the <function_name> directory 
    function_name_dir = joinpath(data_dir, function_name)

    # Create all of the subdirectories
    for ft in function_trends
        for st in surrogate_trends
            # Creates the path for the trend of the function being modeled
            trend_sub_dir = joinpath(function_name_dir, "$(ft)_trend")

            # Creates the path for the trend of the parametric component of the
            # hybrid model
            surrogate_sub_dir = joinpath(trend_sub_dir, "$(st)_trend_surrogate")
            if !isdir(surrogate_sub_dir)
                mkpath(surrogate_sub_dir)
            end

            # Create trend.txt metadata and other metric files for each surrogate
            filenames = ["trend.txt", "gap.csv", "simple_regret.csv", "observations.csv", "minimum_observations.csv"]
            [touch("$(surrogate_sub_dir)/$(fn)") for fn in filenames]
        end
    end

    # Create global_minimizer.txt file for <function_name>'s global minimizer
    touch("$(function_name_dir)/global_minimizer.txt")
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