# Experimental configuration
RANDOM_SEED=1906
RANDOM_RESTARTS=64
NUMBER_OF_TRIALS=50
BAYESIAN_OPTIMIZATION_LOOP_BUDGET=200
SHOULD_OPTIMIZE=1

# Test functions to perform Bayesian Optimization on
function_names=(
  "gramacylee"
  "rastrigin1d"
  "rastrigin4d"
  "ackley1d"
  "ackley2d"
  "ackley3d"
  "ackley4d"
  "ackley5d"
  "ackley8d"
  "ackley10d"
  "ackley16d"
  "rosenbrock"
  "sixhump"
  "braninhoo"
  "hartmann3d"
  "goldsteinprice"
  "beale"
  "easom"
  "styblinskitang1d"
  "styblinskitang2d"
  "styblinskitang3d"
  "styblinskitang4d"
  "styblinskitang10d"
  "bukinn6"
  "crossintray"
  "eggholder"
  "holdertable"
  "schwefel1d"
  "schwefel2d"
  "schwefel3d"
  "schwefel4d"
  "schwefel10d"
  "levyn13"
  "trid1d"
  "trid2d"
  "trid3d"
  "trid4d"
  "trid10d"
  "mccormick"
  "hartmann4d"
  "hartmann6d"
  "bohachevsky"
  "griewank3d"
  "shekel4d"
  "dropwave"
  "griewank1d"
  "griewank2d"
  "levy10d"
)

IFS=','
function_names_joined="${function_names[*]}"
unset IFS

if [ $SHOULD_OPTIMIZE -eq 1 ]; then
  julia ../bayesopt_batch.jl --function-name $function_names_joined --seed $RANDOM_SEED \
    --starts $RANDOM_RESTARTS --trials $NUMBER_OF_TRIALS --budget $BAYESIAN_OPTIMIZATION_LOOP_BUDGET \
    --optimize
else
  julia ../bayesopt_batch.jl --function-name $function_names_joined --seed $RANDOM_SEED \
    --starts $RANDOM_RESTARTS --trials $NUMBER_OF_TRIALS --budget $BAYESIAN_OPTIMIZATION_LOOP_BUDGET
fi