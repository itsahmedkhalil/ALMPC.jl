# [test/runtests.jl]
using ALMPC
using Test
using StaticArrays
using LinearAlgebra
using SparseArrays
using OSQP

# Test scripts
include("foo_test.jl")
include("qp_test.jl")