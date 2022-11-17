# [test/MPC_OSQP_test.jl]

using StaticArrays
using SparseArrays
using LinearAlgebra
using OSQP
using Random
using Test



@testset "MPC OSQP Test" begin 

     # Planar Point Mass Dynamics
function pointmass_dynamics(x,u; mass = 2, damp = 0.3)
     xdot = zero(x) 
     xdot[1] = x[3]
     xdot[2] = x[4]
     xdot[3] = -(damp/mass)*x[3] + u[1]/mass 
     xdot[4] = -(damp/mass)*x[4] + u[2]/mass 
     return xdot
 end

    # Setting up the problem
     dt = 0.05                # Time step [s]
     Tfinal = 1           # Final time [s]
     Nt = Int(Tfinal/dt)+1    # Number of time steps

     Nx = 4  # Number of states
     Nu = 2  # Number of controls

     x0 = [3.0, 7.0, -3.0, -1.0]
     # xfinal = [10.0, 10.0, 0.0, 0.0]
     ueq = zeros(2)

     # Generate reference trajectory
     Xref = ALMPC.nominal_trajectory(x0,Nt,dt)     # Reference trajectory for all states
     Uref = [copy(ueq) for k = 1:Nt]             # Reference inputs 
     tref = range(0,Tfinal, length=Nt)           # Array of timesteps

     mass = 2    # Mass [kg]
     damp = 0.3  # Damping coefficient [N-s/m]

     # Discretized MPC Model dynamics: x_k+1 = Ad*x_k + Bb*u_k
     A = [1.0    0.0     dt                   0.0             ;
          0.0    1.0     0.0                 dt               ;
          0.0    0.0     1-(damp/mass)*dt     0.0             ;
          0.0    0.0     0.0                 1-(damp/mass)*dt]    # State Matrix
     B = zeros(4, 2)                                             # Input Matrix
     B[3,1] = (1/mass)*dt
     B[4,2] = (1/mass)*dt

     # #State Constraints
     # xmin = [-100.0 -100.0]  
     # xmax = [100.0 100.0]    

     # #Control Constraints
     # umin = [-0.5; -0.5]   
     # umax = [0.5; 0.5]    

     # # Constraints input variation with respect to previous sample
     # Dumin = [-2e-1] 
     # Dumax = [2e-1]

     # MPC objective function weights
     Q = Array(10.0*I(Nx));
     R = Array(.01*I(Nu));
     Qf = Array(10.0*I(Nx));

     Nmpc = 51           # MPC Horizon
     mpc1 = ALMPC.OSQPController(Nx, Nu, Nmpc, Q, R, Qf,A, B, length(Xref))
     
     # Provide the reference trajectory
     mpc1.Xref .= Xref
     mpc1.Uref .= Uref
     mpc1.times .= tref

     # Build the sparse QP matrices
     ALMPC.buildQP!(mpc1, A,B,Q,R,Qf, tol=1e-6)
     Xmpc1,Umpc1,tmpc1 = ALMPC.simulate(pointmass_dynamics, x0, mpc1, tf=Nmpc)
     @test norm(Xmpc1[end]) < 1e-6 



end