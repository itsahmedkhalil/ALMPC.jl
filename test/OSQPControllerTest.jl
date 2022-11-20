# [test/MPC_OSQP_test.jl]

using StaticArrays
using SparseArrays
using LinearAlgebra
using OSQP
using Random
using Test


@testset "OSQP Controller Test" begin 
     """
     System Dynamics of Point Mass
     """ 
     mass = 1.0    # Mass [kg]
     damp = 0.1    # Damping coefficient [N-s/m]
     dt = 0.01           # Time step [s]

     # State matrix
     Ad = sparse([   1.0     0.0     dt                   0.0                 ;
                    0.0     1.0     0.0                  dt                  ;
                    0.0     0.0     1-(damp/mass)*dt     0.0                 ;
                    0.0     0.0     0.0                  1-(damp/mass)*dt    ])    

     # Input matrix
     Bd = sparse([   0.0             0.0         ;
                    0.0             0.0         ;
                    (1/mass)*dt     0.0         ;
                    0.0             (1/mass)*dt ])

     # Dynamics of point mass 
     function pointmass_dynamics(x,u; mass = mass, damp = damp)
          xdot = zero(x) 
          xdot[1] = x[3]
          xdot[2] = x[4]
          xdot[3] = -(damp/mass)*x[3] + u[1]/mass 
          xdot[4] = -(damp/mass)*x[4] + u[2]/mass 
          return xdot
     end

     # Number of states (Nx) and inputs (Nu)
     Nx, Nu = size(Bd)

     # Input constraints (x,y)
     umin = [-1, -1]   
     umax = [1, 1]

     # State constraints (x, y, ̇x, ̇y)
     xmin = [-20.0; -20.0; -Inf; -Inf]  
     xmax = [10.0; 10.0; Inf; Inf] 

     # State (Q), input (R), and terminal state (QN) cost matrices
     Q = sparse(0.0*I(Nx))
     R = sparse(.01*I(Nu))
     QN = sparse(1000.0*I(Nx))

     Nmpc = 50   #MPC time horizon [s]  

     # Number of constraints 
     Nd = (Nmpc+1)*(Nx+length(xmin)+length(umin)) - length(umin)

     Tfinal = 50             # Final time [s]
     Nt = Int(Tfinal/dt)+1   # Number of time steps

     x0 = [5.0, 5.0, 0.0, 0.0]   # Initial state (x, y, ̇x, ̇y)
     xf = [-5.0, -5.0, 0.0, 0.0] # Final state (x, y, ̇x, ̇y)
     ueq = zeros(2)              # Equilibrium inputs

     # Generate reference trajectory
     Xref = ALMPC.linear_trajectory(x0,xf,Nt,dt)     # Reference trajectory for all states
     Uref = [copy(ueq) for k = 1:Nt]                 # Reference inputs 
     tref = range(0,Tfinal, length=Nt)               # Array of timesteps

     mpc1 = ALMPC.OSQPController(Nmpc, Q, R, QN, Ad, Bd, length(Xref), Nd)

     # Provide the reference trajectory
     mpc1.Xref .= Xref
     mpc1.Uref .= Uref
     mpc1.times .= tref
     
     # Build the sparse QP matrices
     ALMPC.buildQP!(mpc1, x0, xmin, xmax, umin, umax, tol=1e-6)
     
     Xmpc1,Umpc1,tmpc1 = ALMPC.OSQPSimulate(pointmass_dynamics, x0, mpc1, tf=50)
     
     @test norm(Xmpc1[:,end] - Xref[end]) < 1e-2 
     @test max(Xmpc1[1,end]) < xmax[1]
     @test max(Xmpc1[2,end]) < xmax[2]
     @test max(Xmpc1[3,end]) < xmax[3]
     @test max(Xmpc1[4,end]) < xmax[4]

     @test max(Umpc1[1,end]) < umax[1]
     @test max(Umpc1[2,end]) < umax[2]

     @test min(Xmpc1[1,end]) > xmin[1]
     @test min(Xmpc1[2,end]) > xmin[2]
     @test min(Xmpc1[3,end]) > xmin[3]
     @test min(Xmpc1[4,end]) > xmin[4]

     @test min(Umpc1[1,end]) > umin[1]
     @test min(Umpc1[2,end]) > umin[2]

end