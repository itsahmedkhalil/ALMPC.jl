
get_k(ctrl, t) = searchsortedlast(ctrl.times, t)

function get_control(ctrl, x, t)
    k = get_k(ctrl, t)
    u = ctrl.Uref[k] - ctrl.K*(x - ctrl.Xref[k])
    return u
end

function rk4_step(f,xk,h) 
    xn = zero(xk)
    f1 = f(xk)
    f2 = f(xk + 0.5*h*f1)
    f3 = f(xk + 0.5*h*f2)
    f4 = f(xk + h*f3)
    xn .+= (h/6.0)*(f1 + 2*f2 + 2*f3 + f4)
    return xn
end

function simulate(model, x0, ctrl; tf=ctrl.times[end], dt=1e-2)
    n,m = size(model)
    times = range(0, tf, step=dt)
    N = length(times)
    X = [@SVector zeros(n) for k = 1:N] 
    U = [@SVector zeros(m) for k = 1:N-1]
    X[1] = x0

    tstart = time_ns()
    for k = 1:N-1
        U[k] = get_control(ctrl, X[k], times[k])
        X[k+1] = rk4_step(RK4, model, X[k], U[k], times[k], dt)
    end
    tend = time_ns()
    rate = N / (tend - tstart) * 1e9
    println("Controller ran at $rate Hz")
    return X,U,times
end