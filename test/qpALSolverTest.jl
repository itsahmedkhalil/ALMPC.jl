# [test/qpALSolverTest.jl]

using StaticArrays
using LinearAlgebra
using SparseArrays
using OSQP
using Random

@testset "ALQP Test" begin 
    # Initial guess
    q0=SA[0,1.] 
    v0=SA[1,0.] 
    h=0.01 
    T=3.0
    m=1.0

    M = SA[m 0; 0 m]
    g = SA[0,9.81]
    J = SA[0 1.]
    P = Symmetric(M)
    q = M*(h*g - v0)
    A = @SMatrix zeros(Float64,0,2)
    b = @SVector zeros(Float64,0)
    C = -J*h
    d = J*q0
    a = @MVector zeros(Bool,1)

    λ = @SVector zeros(Float64,0)
    μ = @SVector zeros(Float64,1)
    x = v0

    # Solve
    solver = ALMPC.ALQP(P,q,A,b,C,d,a, 10.0, 10.0)

    @test size(solver.P) == (2,2)
    @test issymmetric(solver.P)
    @test length(solver.q) == 2
    @test size(solver.A) == (0,2)
    @test size(solver.C) == (1,2)
    @test length(solver.d) == 1
    @test length(solver.a) == 1


    xstar, λstar, μstar = ALMPC.solve(solver, x, λ, μ)
    
    # Check optimality conditions
    @test norm(ALMPC.primal_residual(solver, xstar, λstar, μstar)) < 1e-3 
    @test norm(ALMPC.dual_residual(solver, xstar)) < 1e-6    
    @test norm(ALMPC.complimentarity(solver, xstar, λstar, μstar)) < 1e-3  
    

    # Compare with OSQP

    P = zeros(2,2)
    q = zeros(2)
    A = zeros(0,2)
    b = zeros(0)
    C = zeros(1,2)
    d = zeros(1)
    p = length(d)
    
    J = [0 1]
    M = I(2)*m
    g = [0; 9.81]
    q .= M*(h*g .- v0)
    P .= M
    C .= -h*J
    d .= J*q0

    model = OSQP.Model()
    OSQP.setup!(model, P=sparse(P), q=q, A=sparse([A; C]), l=[b; fill(-Inf,p)], u=[b; d],
        eps_abs=1e-6, eps_rel=1e-6, verbose=false)
    res = OSQP.solve!(model)
    @test norm(res.x - xstar) < 1e-3           
    @test norm(res.y - [λstar; μstar]) < 1e-3  
    #@show "Testing done"
end