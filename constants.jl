const RANDOM_ACQUISITION = "Random"
const DOUBLE = 2
const JITTER = 1e-8
const NEWTON_SOLVE_TIME_LIMIT = .1

"""
The amount of space to allocate for our surrogate model in terms of the number of 
observations.
"""
const DEFAULT_CAPACITY = 100