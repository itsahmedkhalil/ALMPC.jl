using StaticArrays
using LinearAlgebra
using SparseArrays


#TODO: 
""" 
    - Add more algos inaddition to the active set:
        - Barrier/Interior-Point method
        - Penalty method
"""

"""
    ALQP

Structure for holding the QP when solving
    min 0.5x'Px + q'x
    s.t.    Ax = b, 
            Cx ≤ d

    a determines whether the inequalities are active 
    ρ is the penalty value applied when constraints are violated
    ϕ is the penalty scaling applied to the penalty value 

The Augmented Lagrangian is defined as:
    L = 0.5x'P'x + q'x + λ'(Ax - b) + μ'(Cx -d) + 0.5ρ||Ax-b||^2 + 0.5ρ||min(0, (Cx - d))||^2

n: number of states
m: number of equality constraints
p: number of inequality constraints 
"""
mutable struct ALQP{n,m,p,nn,mn,pn,T} 
    P::Symmetric{T,SMatrix{n,n,T,nn}}
    q::SVector{n,T}
    A::SMatrix{m,n,T,mn}
    b::SVector{m,T}
    C::SMatrix{p,n,T,pn}
    d::SVector{p,T}
    a::MVector{p,Bool}  
    ρ::T                
    ϕ::T                
end

"""
    active_set!(QP, h, μ)

Determines whether the constraint is active. 

The inequality constraint is defined as:
    Cx ≤ d 

If the equality constraint, h = Cx - d, is being violated, i.e. > 0 
OR if the lagrange multipler, μ, is active, i.e. > 0:
    Then the constrain is active   
"""
function active_set!(opt::ALQP{n,m,p}, h, μ) where {n,m,p}
    for i = 1:p
        opt.a[i] = μ[i] > 0 || h[i] > 0
    end
end

"""
    primal_residual(QP, x, λ, μ)

Returns the derivative of the Lagrangian without the penalties.
    r_p = Px + q + A'λ + C'μ

Checks if derivative of Lagrangian ≈ 0.
"""
function primal_residual(opt::ALQP{n,m,p}, x, λ, μ) where {n,m,p}
    r = opt.P * x + opt.q
    m > 0 && (r += opt.A'λ)
    p > 0 && (r += opt.C'μ)
    return r
end

"""
    dual_residual(QP, x)

Returns the constraint residuals
    g(x) = Ax - b
    h(x) = Cx - d , if > 0, otherwise = 0

Checks if the constraints ≈ 0.
"""
function dual_residual(opt::ALQP, x)
    g = opt.A * x - opt.b
    h = opt.C * x - opt.d
    return [g; max.(0, h)]
end

"""
    complimentarity(QP, x, λ, μ)

Ensures switching between constraints. 
Either the Lagrange multipler μ = 0 or the inequality 
constraint (Cx - d) = 0. 
"""
function complimentarity(opt::ALQP, x, λ, μ)
    return [min.(0, μ); μ .* (opt.C * x - opt.d)]
end

"""
    grad(QP, x, λ, μ)

Computes the gradient of the Augmented Lagrangian. 
"""
function grad(opt::ALQP{n,m,p}, x, λ, μ) where {n,m,p}
    A, C, ρ = opt.A, opt.C, opt.ρ
    grad = opt.P * x + opt.q
    if m > 0
        g = A * x - opt.b
        grad += A'λ + ρ * A'g
    end
    if p > 0
        h = C * x - opt.d
        active_set!(opt, h, μ)
        Iρ = ρ * Diagonal(SVector(opt.a))
        grad += C'μ + C' * (Iρ * h)
    end
    return grad
end

"""
    hess(QP)

Computes the Hessian of the Augmented Lagrangian. 
"""
function hess(opt::ALQP{n,m,p}) where {n,m,p}
    H = opt.P
    A, ρ, C = opt.A, opt.ρ, opt.C
    if m > 0
        H += ρ * (A'A)
    end
    if p > 0
        Iρ = Diagonal(SVector(opt.a))
        H += ρ * (C'Iρ * C)
    end
    return H
end

"""
    dual_update(QP, x, λ, μ)

Updates the Lagrange multipleres, μ and λ after each Newton step. 
"""
function dual_update(opt::ALQP, x, λ, μ)
    ρ = opt.ρ
    λnext = λ + ρ * (opt.A * x - opt.b)
    μnext = max.(0, μ + ρ * (opt.C * x - opt.d))
    return λnext, μnext
end

"""
    inner_solve(QP, x, λ, μ; ϵ, max_iters)

Newton step.
Terminates when error, ϵ, is below a predermined tolerance 
OR when the number of iterations exceed the max_iters. 
"""
function inner_solve(opt::ALQP, x, λ, μ; ϵ=1e-6, max_iters=10)
    for _ = 1:max_iters
        r = grad(opt, x, λ, μ)
        if norm(r) < ϵ
            return x
        end
        H = hess(opt)
        x += -H \ r
    end
    @warn "Inner solve max iterations"
    return x
end

"""
    solve(QP, x, λ, μ; ϵ, max_iters, eps_primal, eps_inner)

Main solver. 
Terminates when both the gradient of the Lagrangian and the constraints 
are met, i.e. when both are ≈ 0 and below the tolerances, eps_grad and 
eps_cons, respectively.
OR when the number of iterations exceed the max_iters. 
"""
function solve(opt::ALQP, x, λ, μ;
    max_iters=10,
    eps_cons=1e-6,
    eps_grad=1e-6)
    for _ = 1:max_iters
        x = inner_solve(opt, x, λ, μ, ϵ=eps_grad, max_iters=max_iters)
        λ, μ = dual_update(opt, x, λ, μ)
        opt.ρ *= opt.ϕ
        if norm(dual_residual(opt, x)) < eps_cons && norm(primal_residual(opt, x, λ, μ)) < eps_grad
            # Return the optimized variables
            return x, λ, μ
        end
    end
    @warn "Outer loop max iterations"
    return x, λ, μ
end