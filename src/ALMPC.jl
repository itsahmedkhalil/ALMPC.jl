module ALMPC

using StaticArrays
using LinearAlgebra
using SparseArrays
using OSQP

greet() = print("Hello World!")

include("foo.jl")
include("qpALSolver.jl")
include("MPC_OSQP.jl")

end # module
