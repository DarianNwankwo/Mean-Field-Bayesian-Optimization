#!/bin/bash

# Experimental configuration
RANDOM_SEED=1906
RANDOM_RESTARTS=256
NUMBER_OF_TRIALS=50
BAYESIAN_OPTIMIZATION_LOOP_BUDGET=100
SHOULD_OPTIMIZE=1

# Array of test function names
function_names=(
  # "rastrigin4d"
  # "ackley2d"
  # "ackley8d"
  # "rosenbrock"
  # "sixhump"
  # "braninhoo"
  # "goldsteinprice"
  # "styblinskitang10d"
  # "schwefel4d"
  "trid4d"
  "mccormick"
  "hartmann6d"
  "griewank3d"
  "shekel4d"
)

> timing.txt

# Loop over each function name and execute the CLI script
for fn in "${function_names[@]}"; do
  (
    echo "Running $fn..."
    start=$(date +%s.%N)
    if [ $SHOULD_OPTIMIZE -eq 1 ]; then
      julia ../bayesopt_batch.jl --function-names "$fn" --seed $RANDOM_SEED \
        --starts $RANDOM_RESTARTS --trials $NUMBER_OF_TRIALS --budget $BAYESIAN_OPTIMIZATION_LOOP_BUDGET --optimize
    else
      julia ../bayesopt_batch.jl --function-names "$fn" --seed $RANDOM_SEED \
        --starts $RANDOM_RESTARTS --trials $NUMBER_OF_TRIALS --budget $BAYESIAN_OPTIMIZATION_LOOP_BUDGET
    fi
    end=$(date +%s.%N)
    elapsed=$(echo "$end - $start" | bc)
    echo "$fn: $elapsed seconds" > "timing_$fn.txt"
  ) # &
done

wait

cat timing_*.txt > timing.txt
rm timing_*.txt