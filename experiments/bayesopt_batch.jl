using ArgParse
import ProgressMeter

CONSTANT_TREND = 1.

@doc raw"""parse_command_line

The parameters we are interested in changing are controlled by modifying the arguments passed to a CLI program.
"""
function parse_command_line(args)
    parser = ArgParseSettings("Bayesian Optimization CLI")

    @add_arg_table! parser begin
        "--seed"
            action = :store_arg
            help = "Seed for random number generation"
            default = 1906
            arg_type = Int
        "--starts"
            action = :store_arg
            help = "Number of random starts for solving the acquisition function (default: 64)"
            default = 64
            arg_type = Int
        "--kernel-starts"
            action = :store_arg
            help = "Number of random starts for learning hyperparameters"
            default = 16
            arg_type = Int
        "--trials"
            action = :store_arg
            help = "Number of trials with a different initial start (default: 50)"
            default = 50
            arg_type = Int
        "--budget"
            action = :store_arg
            help = "Maximum budget for bayesian optimization (default: 200)"
            default = 200
            arg_type = Int
        "--function-names"
            action = :store_arg
            help = "Names of the synthetic function to optimize separated by a comma"
            required = true
        "--optimize"
            action = :store_true
            help = "If set, estimate optimal hyperparameters via MLE"
        "--observation-noise"
            action = :store_arg
            help = "Noise variance of the observations"
            arg_type = Float64
            default = 0.
    end

    parsed_args = parse_args(args, parser)
    return parsed_args
end


include("../bayesian_optimization.jl")



function main()
    cli_args = parse_command_line(ARGS)
    println("Initializing Experimental Design...")

    # Establish the synthetic functions we want to evaluate our algorithms on.
    testfn_payloads = Dict(
        "gramacylee" => (name="gramacylee", fn=TestGramacyLee, args=()),
        "rastrigin1d" => (name="rastrigin1d", fn=TestRastrigin, args=(1)),
        "rastrigin4d" => (name="rastrigin4d", fn=TestRastrigin, args=(4)),
        "ackley1d" => (name="ackley1d", fn=TestAckley, args=(1)),
        "ackley2d" => (name="ackley2d", fn=TestAckley, args=(2)),
        "ackley3d" => (name="ackley3d", fn=TestAckley, args=(3)),
        "ackley4d" => (name="ackley4d", fn=TestAckley, args=(4)),
        "ackley5d" => (name="ackley5d", fn=TestAckley, args=(5)),
        "ackley8d" => (name="ackley8d", fn=TestAckley, args=(8)),
        "ackley10d" => (name="ackley10d", fn=TestAckley, args=(10)),
        "ackley16d" => (name="ackley16d", fn=TestAckley, args=(16)),
        "rosenbrock" => (name="rosenbrock", fn=TestRosenbrock, args=()),
        "sixhump" => (name="sixhump", fn=TestSixHump, args=()),
        "braninhoo" => (name="braninhoo", fn=TestBraninHoo, args=()),
        "hartmann3d" => (name="hartmann3d", fn=TestHartmann3D, args=()),
        "goldsteinprice" => (name="goldsteinprice", fn=TestGoldsteinPrice, args=()),
        "beale" => (name="beale", fn=TestBeale, args=()),
        "easom" => (name="easom", fn=TestEasom, args=()),
        "styblinskitang1d" => (name="styblinskitang1d", fn=TestStyblinskiTang, args=(1)),
        "styblinskitang2d" => (name="styblinskitang2d", fn=TestStyblinskiTang, args=(2)),
        "styblinskitang3d" => (name="styblinskitang3d", fn=TestStyblinskiTang, args=(3)),
        "styblinskitang4d" => (name="styblinskitang4d", fn=TestStyblinskiTang, args=(4)),
        "styblinskitang10d" => (name="styblinskitang10d", fn=TestStyblinskiTang, args=(10)),
        "bukinn6" => (name="bukinn6", fn=TestBukinN6, args=()),
        "crossintray" => (name="crossintray", fn=TestCrossInTray, args=()),
        "eggholder" => (name="eggholder", fn=TestEggHolder, args=()),
        "holdertable" => (name="holdertable", fn=TestHolderTable, args=()),
        "schwefel1d" => (name="schwefel1d", fn=TestSchwefel, args=(1)),
        "schwefel2d" => (name="schwefel2d", fn=TestSchwefel, args=(2)),
        "schwefel3d" => (name="schwefel3d", fn=TestSchwefel, args=(3)),
        "schwefel4d" => (name="schwefel4d", fn=TestSchwefel, args=(4)),
        "schwefel10d" => (name="schwefel10d", fn=TestSchwefel, args=(10)),
        "levyn13" => (name="levyn13", fn=TestLevyN13, args=()),
        "trid1d" => (name="trid1d", fn=TestTrid, args=(1)),
        "trid2d" => (name="trid2d", fn=TestTrid, args=(2)),
        "trid3d" => (name="trid3d", fn=TestTrid, args=(3)),
        "trid4d" => (name="trid4d", fn=TestTrid, args=(4)),
        "trid10d" => (name="trid10d", fn=TestTrid, args=(10)),
        "mccormick" => (name="mccormick", fn=TestMccormick, args=()),
        "hartmann6d" => (name="hartmann6d", fn=TestHartmann6D, args=()),
        "hartmann4d" => (name="hartmann4d", fn=TestHartmann4D, args=()),
        "bohachevsky" => (name="bohachevsky", fn=TestBohachevsky, args=()),
        "griewank3d" => (name="griewank3d", fn=TestGriewank, args=(3)),
        "shekel4d" => (name="shekel4d", fn=TestShekel, args=()),
        "dropwave" => (name="dropwave", fn=TestDropWave, args=()),
        "griewank1d" => (name="griewank1d", fn=TestGriewank, args=(1)),
        "griewank2d" => (name="griewank1d", fn=TestGriewank, args=(2)),
        "levy10d" => (name="levy10d", fn=TestLevy, args=(10)),
    )

    testfn_names = split(cli_args["function-names"], ",")

    # Function trends and surrogate parametric trends
    function_trend_names = [
        "zero_trend",
        "constant_trend",
        "linear_trend",
        "nonlinear_trend"
    ]

    surrogate_trend_names = [
        "zero_trend_surrogate",
        "constant_trend_surrogate",
        "linear_trend_surrogate",
        "nonlinear_trend_surrogate"
    ]

    # Create trend.txt metadata and other metric files for each surrogate
    filenames = ["gaps.csv", "simple_regrets.csv", "observations.csv", "minimum_observations.csv"]
    # Each acquisition function will have metrics associated with them
    # acquisition_functions = [EI(), POI(), LCB()]
    acquisition_functions = [EI()]
    acquisition_function_names = [string(af) for af in acquisition_functions]

    # Create directory to store payload for given test function
    filepath_mappings = Dict()
    for testfn_name in testfn_names
        fpm = create_directory_structure(
            testfn_name, function_trend_names, surrogate_trend_names, filenames, acquisition_function_names
        )
        filepath_mappings[testfn_name] = fpm[testfn_name]
    end

    # Create CSVs for each metric we care to maintain
    metrics_to_collect = ["gaps", "simple_regret"]

    # Get current filepath
    current_directory = dirname(@__FILE__)

    # Surrogate hyperparameters
    kernel_lbs, kernel_ubs = [.1], [5.]
    decision_rule_hyperparameters = [0.]
    hyperparameter_optimizer_starts = generate_initial_guesses(cli_args["kernel-starts"], kernel_lbs, kernel_ubs)
    gaps = zeros(cli_args["budget"] + 1)
    simple_regrets = zeros(cli_args["budget"])
    minimum_observations = zeros(cli_args["budget"])

    println("Beginning Dense Experiments...")
    for testfn_name in testfn_names
        # println("Test Function Being Evaluated: $testfn_name")
        # Exract the current test function from the batch of test functions
        payload = testfn_payloads[testfn_name]
        testfn = payload.fn(payload.args...)
        spatial_lbs, spatial_ubs = get_bounds(testfn)

        # Generate the initial starts for the inner optimizer
        inner_optimizer_starts = generate_initial_guesses(cli_args["starts"], spatial_lbs, spatial_ubs)
        
        # Generate surrogate and function trends to offset the test function with
        surrogate_trends, function_trends, initial_observation_sizes = get_trends(CONSTANT_TREND, testfn.dim)

        # Initialize surrogates with preallocated memory to support the full BO loop
        ios = initial_observation_sizes
        surrogates = [
            Surrogate(
                Matern52(), testfn.dim, cli_args["budget"] + ios[1], EI(), cli_args["observation-noise"]
            ),
            # HybridSurrogate(
            #     Matern52(), surrogate_trends[2], testfn.dim, cli_args["budget"] + ios[2], EI(), cli_args["observation-noise"]
            # ),
            # HybridSurrogate(
            #     Matern52(), surrogate_trends[3], testfn.dim, cli_args["budget"] + ios[3], EI(), cli_args["observation-noise"]
            # ),
            # HybridSurrogate(
            #     Matern52(), surrogate_trends[4], testfn.dim, cli_args["budget"] + ios[4], EI(), cli_args["observation-noise"]
            # )
        ]

        for (k, af) in enumerate(acquisition_functions)
            af_name = acquisition_function_names[k]

            for (i, trend) in enumerate(function_trends)
                # Augment the testfn with a trend
                tfn = function_trend_names[i] == "zero_trend" ? testfn : testfn + trend
                tft_name = function_trend_names[i]

                minimizer_path_prefix = "$current_directory/data/$testfn_name/$tft_name/"
                write_global_minimizer_to_disk(minimizer_path_prefix, tfn)

                for (j, surrogate) in enumerate(surrogates)
                    set_decision_rule!(surrogate, af)
                    st_name = surrogate_trend_names[j]

                    # Extract minimizer from testfunction
                    actual_minimum = tfn(tfn.xopt[1])
                    num_initial_observations = ios[j]

                    # Construct path to directory maintaining current experiments data
                    path_prefix = "$current_directory/data/$testfn_name/$tft_name/$st_name/$af_name/"

                    println("Beginning Randomized Trials: ")
                    for trial in 1:cli_args["trials"]
                        
                        # Gather initial design for our statistical model
                        Xinit = randsample(num_initial_observations, tfn.dim, spatial_lbs, spatial_ubs)
                        yinit = tfn(Xinit) + cli_args["observation-noise"] * randn(num_initial_observations)
                        
                        # Set the correct entries in the preallocated surrogate
                        set!(surrogate, Xinit, yinit)

                        # Perform Bayesian optimization loop
                        surrogate = bayesian_optimize!(
                            surrogate,
                            tfn,
                            spatial_lbs,
                            spatial_ubs,
                            kernel_lbs,
                            kernel_ubs,
                            decision_rule_hyperparameters,
                            inner_optimizer_starts,
                            hyperparameter_optimizer_starts,
                            cli_args["budget"]
                        )

                        # Extract performance metrics
                        observations = get_active_observations(surrogate)
                        get_minimum_observations!(minimum_observations, observations, start=num_initial_observations)
                        update_gaps!(gaps, observations, actual_minimum, start_index=num_initial_observations)
                        update_simple_regrets!(simple_regrets, observations, actual_minimum, start_index=num_initial_observations+1)

                        # Write performance metrics to disk.
                        write_gaps_to_disk(path_prefix, gaps, trial)
                        write_observations_to_disk(path_prefix, observations, trial)
                        write_minimum_observations_to_disk(path_prefix, minimum_observations, trial)
                        write_simple_regrets_to_disk(path_prefix, simple_regrets, trial)

                        # ProgressMeter.next!(progress)
                    end
                    flush(stdout)
                end
            end
        end
    end
end


main()