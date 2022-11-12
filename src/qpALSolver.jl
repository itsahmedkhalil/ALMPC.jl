using StaticArrays
using LinearAlgebra
using SparseArrays

struct QPStruct         
    P::Matrix{Float64}
    q::Vector{Float64}
    A::Matrix{Float64}
    b::Vector{Float64}
    C::Matrix{Float64}
    d::Vector{Float64}
end

function QPStruct(n::Int, m::Int, p::Int)
    QPStruct(zeros(n, n), zeros(n), zeros(m, n), zeros(m), zeros(p, n), zeros(p))
end

mutable struct ALQP{n,m,p,nn,mn,pn,T}
    P::Symmetric{T,SMatrix{n,n,T,nn}}
    q::SVector{n,T}
    A::SMatrix{m,n,T,mn}
    b::SVector{m,T}
    C::SMatrix{p,n,T,pn}
    d::SVector{p,T}
    a::MVector{p,Bool}  # active set
    ρ::T                # penalty value
    ϕ::T                # penalty scaling
end

function ALQP(qp::QPStruct, ρ, ϕ)
    n, m, p = size(qp)
    P = Symmetric(SMatrix{n,n}(qp.P))
    q = SVector{n}(qp.q)
    A = SMatrix{m,n}(qp.A)
    b = SVector{m}(qp.b)
    C = SMatrix{p,n}(qp.C)
    d = SVector{p}(qp.d)
    a = @MVector zeros(Bool, p)
    ALQP(P, q, A, b, C, d, a, ρ, ϕ)
end

function active_set!(opt::ALQP{n,m,p}, h, μ) where {n,m,p}
    for i = 1:p
        opt.a[i] = μ[i] > 0 || h[i] > 0
    end
end

function primal_residual(opt::ALQP{n,m,p}, x, λ, μ) where {n,m,p}
    r = opt.P * x + opt.q
    m > 0 && (r += opt.A'λ)
    p > 0 && (r += opt.C'μ)
end

function dual_residual(opt::ALQP, x)
    g = opt.A * x - opt.b
    h = opt.C * x - opt.d
    return [g; max.(0, h)]
end

function complimentarity(opt::ALQP, x, λ, μ)
    return [min.(0, μ); μ .* (opt.C * x - opt.d)]
end

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
end

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

function dual_update(opt::ALQP, x, λ, μ)
    ρ = opt.ρ
    λnext = λ + ρ * (opt.A * x - opt.b)
    μnext = max.(0, μ + ρ * (opt.C * x - opt.d))
    return λnext, μnext
end

function inner_solve(opt::ALQP, x, λ, μ; ϵ=1e-6, max_iters=10, verbose=1)
    for i = 1:max_iters
        r = grad(opt, x, λ, μ)
        if norm(r) < ϵ
            return x
        end
        H = hess(opt)
        dx = -H \ r
        x += dx
    end
    @warn "Inner solve max iterations"
    return x
end

function solve(opt::ALQP, x, λ, μ;
    max_iters=10,
    eps_primal=1e-6,
    eps_inner=1e-6,
    verbose=0)
    for _ = 1:max_iters
        x = inner_solve(opt, x, λ, μ, ϵ=eps_inner, max_iters=max_iters, verbose=verbose)
        # Use the new solution to update the dual variables
        λ, μ = dual_update(opt, x, λ, μ)
        # TODO: update the penalty parameter
        opt.ρ *= opt.ϕ
        if norm(dual_residual(opt, x)) < eps_primal && norm(primal_residual(opt, x, λ, μ)) < eps_inner
            # Return the optimized variables
            return x, λ, μ
        end
    end
    @warn "Outer loop max iterations"
    return x, λ, μ
end