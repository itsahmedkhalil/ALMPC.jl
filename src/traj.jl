using StaticArrays
using SparseArrays
using LinearAlgebra
using OSQP

# Create a linear trajectory from x0 to xN
function nominal_trajectory(x0,N,dt)
    Xref = [zero(x0) for _ = 1:N]
    # Design a trajectory that linearly interpolates from x0 to the origin
    for k = 1:N
        Xref[k][1:2] .= (N-k)/(N-1)*x0[1:2]
    end
    for k = 1:N-1
        Xref[k][3:4] .= (Xref[2][1:2] - Xref[1][1:2])/dt
    end
    return SVector{4}.(Xref)
end