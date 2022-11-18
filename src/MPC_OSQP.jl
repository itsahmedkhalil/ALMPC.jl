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
function OSQPController(N::Integer, Q::Matrix, R::Matrix, Qf::Matrix, Ad::Matrix, Bd::Matrix, Nref::Integer=N, Nd::Integer=(N-1)*n)
    n, m = size(Bd)
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

get_k(ctrl, t) = searchsortedlast(ctrl.times, t)


"""
    buildQP!(ctrl, A,B,Q,R,Qf; kwargs...)

Builds the QP matrices `P` and `A` for the MPC problem. Note that these matrices
should be constant between MPC iterations.

Any keyword arguments will be passed to `initialize_solver!`.
"""
function buildQP!(ctrl::MPCController, A,B,Q,R,Qf,lb,ub; kwargs...)
    if isconstrained(ctrl)
        buildQP_constrained!(ctrl::MPCController, A,B,Q,R,Qf,lb,ub; kwargs...)
    else
        buildQP_unconstrained!(ctrl::MPCController, A,B,Q,R,Qf; kwargs...)
    end
end

"""
    updateQP!(ctrl::MPCController, x, time)

Updates the vectors in the QP problem for the current state `x` and time `time`.
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

Initializes the internal solver once the QP matrices are initialized in the 
controller.
"""
function initialize_solver!(ctrl::MPCController{OSQP.Model}; tol=1e-6, verbose=false)
    OSQP.setup!(ctrl.solver, P=ctrl.P, q=ctrl.q, A=ctrl.A, l=ctrl.lb, u=ctrl.ub, 
        verbose=verbose, eps_rel=tol, eps_abs=tol, polish=1)
end

"""
    get_control(ctrl::MPCController, x, t)

Gets the control from the MPC solver by solving the QP. Change!
"""
function get_control(ctrl::MPCController{OSQP.Model}, x, time)
    # Update the QP
    updateQP!(ctrl, x, time)
    OSQP.update!(ctrl.solver, q=ctrl.q, l=ctrl.lb, u=ctrl.ub)

    # Solve QP
    results = OSQP.solve!(ctrl.solver)
    Δu = results.x[1:ctrl.Nu]
    #@show Δu
    
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

"""
    buildQP!(ctrl, A,B,Q,R,Qf; kwargs...)

Builds the QP matrices `P` and `A` for the MPC problem. Note that these matrices
should be constant between MPC iterations.

Any keyword arguments will be passed to `initialize_solver!`.
"""
function buildQP_constrained!(ctrl::MPCController,A,B,Q,R,Qf,lb,ub; kwargs...)
    # TODO: Implement this method to build the QP matrices
    
    Nu = length(ctrl.Uref[1]) # Number of Inputs
    Nx = length(ctrl.Xref[1]) # Number of States
    Nh = ctrl.Nmpc - 1 # Time horizon 
    Nd = length(ctrl.lb) # Number of constraints

    
    H = sparse([kron(Diagonal(I,Nh-1),[R zeros(Nu,Nx); zeros(Nx,Nu) Q]) zeros((Nx+Nu)*(Nh-1), Nx+Nu); zeros(Nx+Nu,(Nx+Nu)*(Nh-1)) [R zeros(Nu,Nx); zeros(Nx,Nu) Qf]])

    C = sparse([[B -I zeros(Nx,(Nh-1)*(Nu+Nx))]; zeros(Nx*(Nh-1),Nu) [kron(Diagonal(I,Nh-1), [A B]) zeros((Nh-1)*Nx,Nx)] + [zeros((Nh-1)*Nx,Nx) kron(Diagonal(I,Nh-1),[zeros(Nx,Nu) Diagonal(-I,Nx)])]])
    
    U = kron(Diagonal(I,Nh), [zeros(Nu,Nx) I])      #Matrix that picks out all u
    
    # for k = 1:Nx
    #     x = zeros(1,Nu+Nx)
    #     x[k+Nu] = 1
    #     y = kron(Diagonal(I,Nh), x) # Matrix that picks out an x value
    #     C = [C;y]
    # end

    umin = [-0.5; -0.5]   
    umax = [0.5; 0.5]   

    lb = [zeros(Nx*Nh); kron(ones(Nh), umin)]
    ub = [zeros(Nx*Nh); kron(ones(Nh), umax)] 

    D = [C;U]

    if Nd == Nh*Nx
        D = C
        ub = zero(ctrl.ub)
        lb = zero(ctrl.lb)
    end

    lb = lb[1:Nd]
    ub = ub[1:Nd]

    ctrl.P .= H
    ctrl.A .= D
    ctrl.ub .= ub
    ctrl.lb .= lb 


    # Initialize the included solver
    #    If you want to use your QP solver, you should write your own
    #    method for this function
    initialize_solver!(ctrl; kwargs...)
    return nothing
end

"""
    update_QP!(ctrl::MPCController, x, time)

Update the vectors in the QP problem for the current state `x` and time `time`.
This should update `ctrl.q`, `ctrl.lb`, and `ctrl.ub`.
"""
function updateQP_constrained!(ctrl::MPCController, x, time)
    t = get_k(ctrl, time)
    
    Nt = ctrl.Nmpc-1             # horizon
    Nx = length(ctrl.Xref[1])    # number of states
    Nu = length(ctrl.Uref[1])    # number of controls
    Q = ctrl.Q
    R = ctrl.R
    Qf = ctrl.Qf
    A = ctrl.Ad
    B = ctrl.Bd
    
    # Update QP problem
    b = ctrl.q

    # Update lb and ub
    lb = ctrl.lb
    ub = ctrl.ub
    xref = ctrl.Xref
    xeq = xref[end]
    N = length(ctrl.Xref)
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