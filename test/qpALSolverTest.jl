# [test/qpALSolverTest.jl]

using StaticArrays
using LinearAlgebra
using SparseArrays
using OSQP
using Random

Random.seed!(1234);

# ALQP
M = @SMatrix rand(3,3)
P = Symmetric(M'M)
q = @SVector rand(3)
A = @SMatrix rand(1,3)
b = @SVector rand(1) 
C = @SMatrix rand(3,3)
d = @SVector rand(3)
a = @MVector zeros(Bool,3)
λ = @SVector zeros(Float64,1)
μ = @SVector zeros(Float64,3)
x = zeros(3)

# Solve
solver = ALMPC.ALQP(P,q,A,b,C,d,a, 10.0, 10.0)
xstar, λstar, μstar = ALMPC.solve(solver, x, λ, μ)

println("Expect 2 warnings")
~,~,~ = ALMPC.solve(solver, x, λ, μ, max_iters=1)

# OSQP
P = Symmetric(M'M)
qO = zeros(3)
qO .= q
AO = zeros(1,3)
AO .= A
bO = zeros(1)
bO .= b
CO = zeros(3,3)
CO .= C
dO = zeros(3)
dO .= d
p = length(dO)

model = OSQP.Model()
OSQP.setup!(model, P=sparse(P), q=qO, A=sparse([AO; CO]), l=[bO; fill(-Inf,p)], u=[bO; dO],
    eps_abs=1e-6, eps_rel=1e-6, verbose=false)
res = OSQP.solve!(model)


@testset "ALQP Test" begin 
    @test size(solver.P) == (3,3)
    @test issymmetric(solver.P)
    @test length(solver.q) == 3
    @test size(solver.A) == (1,3)
    @test size(solver.C) == (3,3)
    @test length(solver.d) == 3
    @test length(solver.a) == 3

    # Check optimality conditions
    @test norm(ALMPC.primal_residual(solver, xstar, λstar, μstar)) < 1e-3 
    @test norm(ALMPC.dual_residual(solver, xstar)) < 1e-6    
    @test norm(ALMPC.complimentarity(solver, xstar, λstar, μstar)) < 1e-3  

    # Compare with OSQP
    @test norm(res.x - xstar) < 1e-3           
    @test norm(res.y - [λstar; μstar]) < 1e-3  
end