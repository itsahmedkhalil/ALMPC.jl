using StaticArrays
using LinearAlgebra
using SparseArrays
using OSQP

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

Generates an `MPCController` that uses OSQP to solve the QP.
Initializes the controller with matrices consistent with `n` states,
`m` controls, and an MPC horizon of `N`, and `Nd` constraints. 

Uses `Nref` to initialize a reference trajectory whose length may differ from the 
horizon length.
"""
function OSQPController(N::Integer, Q::SparseMatrixCSC{Float64, Int64}, R::SparseMatrixCSC{Float64, Int64}, Qf::SparseMatrixCSC{Float64, Int64}, Ad::SparseMatrixCSC{Float64, Int64}, Bd::SparseMatrixCSC{Float64, Int64}, Nref::Integer=N, Nd::Integer=(N-1)*n)
    n, m = size(Bd)         # Number of states (n), number of inputs (m)
    Np = (N)*(n+m) + n      # Number of primals
    P = spzeros(Np,Np)      # Quadtratic cost matrix
    q = zeros(Np)           # Linear cost vector
    A = spzeros(Nd,Np)      # Constraint matrix
    lb = zeros(Nd)          # Lower constraints vector
    ub = zeros(Nd)          # Upper constraints vector
    Xref = [zeros(n) for _ = 1:Nref]    # Reference state trajectory
    Uref = [zeros(m) for _ = 1:Nref]    # Reference input trajectory
    tref = zeros(Nref)                  # Reference time array
    solver = OSQP.Model()               # Setup OSQP solver
    MPCController{OSQP.Model}(P,q, A,lb,ub, N, solver, Xref, Uref, tref, n, m, Q, R, Qf, Ad, Bd)
end

# Find the current iteration, k
get_k(ctrl, t) = searchsortedlast(ctrl.times, t)


"""
    initialize_solver!(ctrl::MPCController; kwargs...)

Initializes the internal solver once the QP matrices are initialized in the 
controller.
"""
function initialize_solver!(ctrl::MPCController{OSQP.Model}; tol=1e-6, verbose=false)
    OSQP.setup!(ctrl.solver, P=ctrl.P, q=ctrl.q, A=ctrl.A, l=ctrl.lb, u=ctrl.ub, 
        verbose=verbose, eps_rel=tol, eps_abs=tol, polish=1)
end

"""
    get_control(ctrl::MPCController, x, t)

Gets the control from the MPC solver by solving the QP. Change for ALQP!
"""
function get_control(ctrl::MPCController{OSQP.Model}, x, time)
    Nu = ctrl.Nu    # Number of Inputs
    Nx = ctrl.Nx    # Number of States
    N = ctrl.Nmpc   # MPC time horizon [s]

    # Update the QP
    updateQP!(ctrl, x, time)  
    OSQP.update!(ctrl.solver, q=ctrl.q, l=ctrl.lb, u=ctrl.ub)

    # Solve QP
    res = OSQP.solve!(ctrl.solver)
    Δu = res.x[(N+1)*Nx+1:(N+1)*Nx+Nu]
    
    k = get_k(ctrl, time)
    return ctrl.Uref[k] + Δu 
end

"""
    buildQP!(ctrl, A,B,Q,R,Qf; kwargs...)

Builds the QP matrices `P` and `A` for the MPC problem. Note that these matrices
should be constant between MPC iterations.

Any keyword arguments will be passed to `initialize_solver!`.
"""
function buildQP!(ctrl::MPCController,x0,xmin,xmax,umin,umax; kwargs...)    
    Nu = ctrl.Nu    # Number of Inputs
    Nx = ctrl.Nx    # Number of States
    N = ctrl.Nmpc   # MPC time horizon [s]
    Ad = ctrl.Ad    # Discrete state transition matrix
    Bd = ctrl.Bd    # Discrete input transition matrix
    R = ctrl.R      # Input cost matrix  
    Q = ctrl.Q      # State cost matrix
    Qf = ctrl.Qf    # Terminal state cost matrix

    xref = ctrl.Xref    
    xf = xref[end]  # Terminal state

    # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
    # - quadratic objective
    ctrl.P .= blockdiag(kron(I(N), Q), Qf, kron(I(N), R))
    ctrl.q .= [kron(ones(N), -Q*xf); -Qf*xf; zeros(N*Nu)]

    Ax = kron(I(N+1), -I(Nx)) + kron(diagm(-1 => ones(N)), Ad)
    Bu = kron([zeros(1, N); I(N)], Bd)
    Aeq = [Ax Bu]
    leq = [-x0; zeros(N*Nx)]
    ueq = leq
    
    # - input and state constraints
    Aineq = I((N+1)*Nx + N*Nu)
    lineq = [kron(ones(N+1), xmin); kron(ones(N), umin)]
    uineq = [kron(ones(N+1), xmax); kron(ones(N), umax)]
    
    # - OSQP constraints
    ctrl.A .= [Aeq; Aineq]
    ctrl.lb .= [leq; lineq]
    ctrl.ub .= [ueq; uineq]
    
    initialize_solver!(ctrl; kwargs...)
    return nothing
end

"""
    update_QP!(ctrl::MPCController, x, time)

Update the vectors in the QP problem for the current state `x` and time `time`.
This should update `ctrl.q`, `ctrl.lb`, and `ctrl.ub`.
"""
function updateQP!(ctrl::MPCController, x, time)
    t = get_k(ctrl, time)
    
    Nt = ctrl.Nmpc                  # MPC time horizon
    Nx = length(ctrl.Xref[1])       # Number of states
    Nu = length(ctrl.Uref[1])       # Number of inputs
    Q = ctrl.Q                      # State cost matrix
    Qf = ctrl.Qf                    # Terminal state cost matrix
    A = ctrl.Ad                     # Discrete state transition matrix

    # Update QP problem
    b = ctrl.q

    # Update lb and ub
    lb = ctrl.lb
    ub = ctrl.ub
    xref = ctrl.Xref
    xeq = xref[end]
    N = length(ctrl.Xref)

    # Apply current state and terminal state error to cost
    for t_h = 1:(Nt-1)
        if (t+t_h) <= N
            b[(Nu+(t_h-1)*(Nx+Nu)).+(1:Nx)] .= -Q*(xref[t+t_h] - xeq)
        else
            b[(Nu+(t_h-1)*(Nx+Nu)).+(1:Nx)] .= -Q*(xref[end] - xeq)
        end
    end
    if (t+Nt) <= N
        b[(Nu+(Nt-1)*(Nx+Nu)).+(1:Nx)] .= -Qf*(xref[t+Nt] - xeq)
    else
        b[(Nu+(Nt-1)*(Nx+Nu)).+(1:Nx)] .= -Qf*(xref[end] - xeq)
    end

    # Update the initial condition
    lb[1:Nx] .= -A*(x - xeq)
    ub[1:Nx] .= -A*(x - xeq)

    return nothing
end