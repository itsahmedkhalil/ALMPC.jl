using StaticArrays
using SparseArrays
using LinearAlgebra
using OSQP

#TODO: 
""" 
    - Documentation
    - Add more integrators 
    - Make the integrator choice variable
    - @SVector for optimization
    - printing rate - optional!
"""

# Classic RK4 integration with zero-order hold on u
function rk4(f,x,u,h)
    f1 = f(x, u)
    f2 = f(x + 0.5*h*f1, u)
    f3 = f(x + 0.5*h*f2, u)
    f4 = f(x + h*f3, u)
    return x + (h/6.0)*(f1 + 2*f2 + 2*f3 + f4)
end

# Simulating the problem
function simulate(f, x0, ctrl; tf=2.0, dt=0.01)
    Nx = ctrl.Nx
    Nu = ctrl.Nu
    times = range(0, tf, step=dt)
    N = length(times)
    X = zeros(length(x0),N)
    U = zeros(Nu,N-1)
    X[:,1] .= x0
    

    tstart = time_ns()
    for k = 1:N-1
        U[:,k] = get_control(ctrl, X[:,k], times[k]) 
        # U[k] = max.(min.(umax, U[k]), umin)
        X[:,k+1] = rk4(f, X[:,k], U[:,k],dt)
    end
    tend = time_ns()
    rate = N / (tend - tstart) * 1e9
    println("Controller ran at $rate Hz")
    return X,U,times
end
