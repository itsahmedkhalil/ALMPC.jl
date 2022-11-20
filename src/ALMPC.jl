module ALMPC

using StaticArrays
using LinearAlgebra
using SparseArrays
using OSQP

include("foo.jl")
include("qpALSolver.jl")
include("OSQPController.jl")
include("simulation.jl")
include("trajectory.jl")

end # module
