[![CI](https://github.com/itsahmedkhalil/ALMPC.jl/actions/workflows/CI.yml/badge.svg?branch=codecov)](https://github.com/itsahmedkhalil/ALMPC.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/itsahmedkhalil/ALMPC.jl/branch/main/graph/badge.svg?token=QVT2BL4QDR)](https://codecov.io/gh/itsahmedkhalil/ALMPC.jl)
# ALMPC.jl

This is a Julia package intended to solve convex MPC problems using a Primal-Dual Augmented Lagrangian QP solver.
## Installation

Open the Julia REPL, enter the package manager using `]`, and run the following command to clone the code
```bash
    dev https://github.com/itsahmedkhalil/ALMPC.jl.git
```

A folder called ALMPC should be created and it should contain the repo. Make changes and push your code.

## Example with 2D Point Mass 

For a comprehensive example, please refer to this [file](https://github.com/itsahmedkhalil/ALMPC.jl/blob/main/examples/OSQPMPC.ipynb).
### Define the Dynamical System

```
"""
    System Dynamics of Point Mass
""" 
const mass = 1.0    # Mass [kg]
const damp = 0.1    # Damping coefficient [N-s/m]
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
```

### Setup the Problem Formulation 

```
# Number of states (Nx) and inputs (Nu)
Nx, Nu = size(Bd)

# Input constraints (x,y)
umin = [-1, -1]   
umax = [1, 1]

# State constraints (x, y, ̇x, ̇y)
xmin = [-20.0; -20.0; -Inf; -Inf]  
xmax = [10.0; 10.0; Inf; Inf] 

# State (Q), input (R), and terminal state (QN) cost matrices
Q = sparse(1.0*I(Nx))
R = sparse(.01*I(Nu))
QN = sparse(10.0*I(Nx))

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
```

### Build and Solve the QP as MPC

```
# Generate an MPC Struct
mpc1 = ALMPC.ALQPController(Nmpc, Q, R, QN, Ad, Bd, length(Xref), Nd)

# Provide reference state, input, and time trajectories
mpc1.Xref .= Xref
mpc1.Uref .= Uref
mpc1.times .= tref

# Build the sparse QP matrices
ALMPC.buildALQP!(mpc1, x0, xmin, xmax, umin, umax, tol=1e-6)

# Simulate the MPC problem. Returns states and inputs. 
Xmpc1,Umpc1,tmpc1 = ALMPC.ALQPSimulate(pointmass_dynamics, x0, mpc1, tf=50)
```

