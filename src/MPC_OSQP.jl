using StaticArrays
using LinearAlgebra
using SparseArrays
using OSQP
using RobotDynamics

 
# const dt = 0.1  # time step (s)
# const tf = 25   # time horizon (s)
# const N = Int(tf / dt) + 1

# const h = 0.05

# const mass = 1.0
# const damp = 1.0

# # Discretized MPC Model dynamics: x_k+1 = Ad*x_k + Bb*u_k
# const A = [1.0    0.0     h                   0.0             ;
#      0.0    1.0     0.0                 h               ;
#      0.0    0.0     1-(damp/mass)*h     0.0             ;
#      0.0    0.0     0.0                 1-(damp/mass)*h] # State Matrix

# const B =   [   0.0 0.0;
#                 0.0 0.0;
#                 (1/mass)*h 0.0;
#                 0.0 (1/mass)*h;]

"""
    MPCController

An MPC controller that uses a solver of type `S` to solve a QP at every iteration.

It will track the reference trajectory specified by `Xref`, `Uref` and `times` 
with an MPC horizon of `Nmpc`. It will track the terminal reference state if 
the horizon extends beyond the reference horizon.
"""
struct MPCController{S}
    P::SparseMatrixCSC{Float64,Int}
    q::Vector{Float64}
    A::SparseMatrixCSC{Float64,Int}
    lb::Vector{Float64}
    ub::Vector{Float64}
    Nmpc::Int
    solver::S
    Xref::Vector{Vector{Float64}}
    Uref::Vector{Vector{Float64}}
    times::Vector{Float64}
    Nx::Int
    Nu::Int
    Q::SparseMatrixCSC{Float64,Int}
    R::SparseMatrixCSC{Float64,Int}
    Qf::SparseMatrixCSC{Float64,Int}
    Ad::SparseMatrixCSC{Float64,Int}
    Bd::SparseMatrixCSC{Float64,Int}
end

"""
    OSQPController(n,m,N,Nref,Nd)

Generate an `MPCController` that uses OSQP to solve the QP.
Initializes the controller with matrices consistent with `n` states,
`m` controls, and an MPC horizon of `N`, and `Nd` constraints. 

Use `Nref` to initialize a reference trajectory whose length may differ from the 
horizon length.
"""
function OSQPController(n::Integer, m::Integer, N::Integer, Q, R, Qf , Ad, Bd, Nref::Integer=N, Nd::Integer=(N-1)*n)
    Np = (N-1)*(n+m)   # number of primals
    P = spzeros(Np,Np)
    q = zeros(Np)
    A = spzeros(Nd,Np)
    lb = zeros(Nd)
    ub = zeros(Nd)
    Xref = [zeros(n) for _ = 1:Nref]
    Uref = [zeros(m) for _ = 1:Nref]
    tref = zeros(Nref)
    solver = OSQP.Model()
    MPCController{OSQP.Model}(P,q, A,lb,ub, N, solver, Xref, Uref, tref, n, m, Q, R, Qf, Ad, Bd)
end

isconstrained(ctrl::MPCController) = length(ctrl.lb) != (ctrl.Nmpc - 1) * ctrl.Nx

"""
    buildQP!(ctrl, A,B,Q,R,Qf; kwargs...)

Build the QP matrices `P` and `A` for the MPC problem. Note that these matrices
should be constant between MPC iterations.

Any keyword arguments will be passed to `initialize_solver!`.
"""
function buildQP!(ctrl::MPCController, A,B,Q,R,Qf; kwargs...)
    if isconstrained(ctrl)
        buildQP_constrained!(ctrl::MPCController, A,B,Q,R,Qf; kwargs...)
    else
        buildQP_unconstrained!(ctrl::MPCController, A,B,Q,R,Qf; kwargs...)
    end
end

"""
    updateQP!(ctrl::MPCController, x, time)

Update the vectors in the QP problem for the current state `x` and time `time`.
This should update `ctrl.q`, `ctrl.lb`, and `ctrl.ub`.
"""
function updateQP!(ctrl::MPCController, x, time)
    if isconstrained(ctrl)
        updateQP_constrained!(ctrl, x, time)
    else
        updateQP_unconstrained!(ctrl, x, time)
    end
end


"""
    initialize_solver!(ctrl::MPCController; kwargs...)

Initialize the internal solver once the QP matrices are initialized in the 
controller.
"""
function initialize_solver!(ctrl::MPCController{OSQP.Model}; tol=1e-6, verbose=false)
    OSQP.setup!(ctrl.solver, P=ctrl.P, q=ctrl.q, A=ctrl.A, l=ctrl.lb, u=ctrl.ub, 
        verbose=verbose, eps_rel=tol, eps_abs=tol, polish=1)
end

"""
    get_control(ctrl::MPCController, x, t)

Get the control from the MPC solver by solving the QP. 
If you want to use your own QP solver, you'll need to change this
method.
"""
function get_control(ctrl::MPCController{OSQP.Model}, x, time)
    # Update the QP
    updateQP!(ctrl, x, time)
    OSQP.update!(ctrl.solver, q=ctrl.q, l=ctrl.lb, u=ctrl.ub)

    # Solve QP
    results = OSQP.solve!(ctrl.solver)
    Δu = results.x[1:2]
    
    k = get_k(ctrl, time)
    return ctrl.Uref[k] + Δu 
end


function buildQP_unconstrained!(ctrl::MPCController, A,B,Q,R,Qf; kwargs...)
    # TODO: Implement this method to build the QP matrices
    Nu = ctrl.Nu # Number of Inputs
    Nx = ctrl.Nx # Number of States
    Nh = ctrl.Nmpc - 1 # Time horizon 
    Nd = length(ctrl.lb) # Number of constraints
    n = Nx
    Q = ctrl.Q
    R = ctrl.R
    Qf = ctrl.Qf
    H = sparse([kron(Diagonal(I,Nh-1),[R zeros(Nu,Nx); zeros(Nx,Nu) Q]) zeros((Nx+Nu)*(Nh-1), Nx+Nu); zeros(Nx+Nu,(Nx+Nu)*(Nh-1)) [R zeros(Nu,Nx); zeros(Nx,Nu) Qf]])
    C = sparse([[B -I zeros(Nx,(Nh-1)*(Nu+Nx))]; zeros(Nx*(Nh-1),Nu) [kron(Diagonal(I,Nh-1), [A B]) zeros((Nh-1)*Nx,Nx)] + [zeros((Nh-1)*Nx,Nx) kron(Diagonal(I,Nh-1),[zeros(Nx,Nu) Diagonal(-I,Nx)])]])
    
    lb = [zeros(Nx*Nh)]
    ub = [zeros(Nx*Nh)] 

    if Nd == Nh*n
        ub = zero(ctrl.ub)
        lb = zero(ctrl.lb)
    end

    ctrl.P .= H
    ctrl.A .= C
    ctrl.ub .= ub
    ctrl.lb .= lb 

    # Initialize the included solver
    #    If you want to use your QP solver, you should write your own
    #    method for this function
    initialize_solver!(ctrl; kwargs...)
    return nothing
end


function updateQP_unconstrained!(ctrl::MPCController, x, time)
    t = get_k(ctrl, time) # Get the time index corresponding to time 'time'

    Nu = ctrl.Nu # Number of Inputs
    Nx = ctrl.Nx # Number of States
    Nh = ctrl.Nmpc - 1 # Time horizon 
    Nd = length(ctrl.lb) # Number of constraints
    Q = ctrl.Q
    R = ctrl.R
    Qf = ctrl.Qf
    A = ctrl.Ad
    B = ctrl.Bd


    #Update QP 
    b = ctrl.q
    lb = ctrl.lb # Lower bound 
    ub = ctrl.ub # Upper bound
    xref = ctrl.Xref # Reference trajectory
    xf = xref[end] # Final state
    N = length(ctrl.Xref) # Total time duration 

    for t_h = 1:(Nh-1) # Take timesteps
        if (t+t_h) <= N # If not at the final timestep
            b[(Nu+(t_h-1)*(Nx+Nu)).+(1:Nx)] .= -Q*(xref[t+t_h] - xf)
        else
            b[(Nu+(t_h-1)*(Nx+Nu)).+(1:Nx)] .= -Q*(xref[end] - xf)
        end
    end

    if (t+Nh) <= N
        b[(Nu+(Nh-1)*(Nx+Nu)).+(1:Nx)] .= -Qf*(xref[t+Nh] - xf)
    else
        b[(Nu+(Nh-1)*(Nx+Nu)).+(1:Nx)] .= -Qf*(xref[end] - xf)
    end
    
    # Update the initial condition
    lb[1:Nx] .= -A*(x - xf)
    ub[1:Nx] .= -A*(x - xf)

    #@show Nx, Nh, length(lb)
    return nothing
end

get_k(ctrl, t) = searchsortedlast(ctrl.times, t)
