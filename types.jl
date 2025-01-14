abstract type AbstractDecisionRule end
abstract type AbstractKernel end
abstract type StationaryKernel <: AbstractKernel end
abstract type NonStationaryKernel <: AbstractKernel end
abstract type AbstractSurrogate end
abstract type AbstractParametricRepresentation end
abstract type ParametricRepresentation <: AbstractParametricRepresentation end