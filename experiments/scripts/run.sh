experiment_configurations=(
    # Configuration short long run experiments
    "--function-name ackley1d"
    "--function-name ackley5d"
    "--function-name braninhoo"
    "--function-name hartmann6d"
    "--function-name sixhump"
    "--function-name levy10d"
    "--function-name goldsteinprice"
    "--function-name griewank3d"
)

for config in "${experiment_configurations[@]}"; do
  julia ../bayesopt.jl $config
done