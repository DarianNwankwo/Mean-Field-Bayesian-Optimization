using ArgParse


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
        "--function-name"
            action = :store_arg
            help = "Name of the synthetic function to optimize"
            required = true
        "--optimize"
            action = :store_true
            help = "If set, estimate optimal hyperparameters via MLE"
    end

    parsed_args = parse_args(args, parser)
    return parsed_args
end


cli_args = parse_command_line(ARGS)

include("../bayesian_optimization.jl")


function main()
end


main()