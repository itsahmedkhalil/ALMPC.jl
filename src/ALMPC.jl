module ALMPC

using StaticArrays
using LinearAlgebra
using SparseArrays
using OSQP

include("foo.jl")
include("qpALSolver.jl")
include("MPC_OSQP.jl")
include("simulation.jl")
include("trajectory.jl")

end # module
