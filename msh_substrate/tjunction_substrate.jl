# T-junction array on the substrate (Julia side of tjunction_substrate.py): geometry, texture, exchange_12, and H(t) MPOs.
include(joinpath(@__DIR__, "substrate_model.jl"))

const NLEG, SPACING = 16, 32
const ARM_ANG = (0.0, π, π / 2)
const ARM_ORIG = ((0, 0), (-1, 0), (0, 1))
const ARM_DIR = ((1, 0), (-1, 0), (0, 1))
center_x(c, ncell) = Int((c - (ncell + 1) / 2) * SPACING)
array_pos(c, s, n, ncell) = (center_x(c, ncell) + ARM_ORIG[s][1] + n * ARM_DIR[s][1], ARM_ORIG[s][2] + n * ARM_DIR[s][2])
tlabels(ncell) = Dict(array_pos(c, s, n, ncell) => (c, s, n) for c in 1:ncell for s in 1:3 for n in 0:NLEG-1)
"Lattice position of the array point (0, 0): centred in x, horizontal row at Ny/2 (as origin() in Python)."
torigin(g::Substrate) = (Nx(g) ÷ 2, Ny(g) ÷ 2)
"Exchange sites Dict (ix, iy) => S for the texture f(c, s, n): θ = (−1)^(x+y) (π/2) f, φ = θ_arm + π/2."
function tspins(f, g::Substrate; ncell=1)
    ox, oy = torigin(g)
    Dict((ox + x, oy + y) => (θ = (-1)^(x + y) * (π / 2) * f(c, s, n); φ = ARM_ANG[s] + π / 2;
                              (cos(φ) * sin(θ), sin(φ) * sin(θ), cos(θ))) for ((x, y), (c, s, n)) in tlabels(ncell))
end

# exchange_12 (L1 = L2 = 14, rot_T = 200, step_T = 0.1), as in tjunctions_model.py
Base.@kwdef struct Exchange12
    L1::Int = 14; L2::Int = 14; rot_T::Float64 = 200.0; step_T::Float64 = 0.1
end
duration(p::Exchange12) = (p.step_T * (4p.L1 + 2p.L2 + 8) + 1) * p.rot_T
function f_horizontal_12(p, i, t)
    τ, s, L1, L2 = t / p.rot_T, p.step_T, p.L1, p.L2
    0 <= i <= L1 && return smoothstep(τ - s * (i + L1 + L2 + 2)) - smoothstep(τ - s * (L1 - i + 3L1 + 2L2 + 8))
    i == L1 + 1 && return 0.0
    L1 + 2 <= i <= 2L1 + 2 && return smoothstep(τ - s * (2L1 + 2 - i)) - smoothstep(τ - s * (i + L1 + L2 + 2))
    return 1.0
end
f_vertical_12(p, k, t) = (τ = t / p.rot_T; s = p.step_T;
                          1 - smoothstep(τ - s * (k + p.L1 + 2)) + smoothstep(τ - s * (p.L2 - k + 3p.L1 + p.L2 + 5)))
f_braid_12(p, t) = (c, s, n) -> s == 1 ? f_horizontal_12(p, p.L1 + 1 + n, t) :
                                s == 2 ? f_horizontal_12(p, max(p.L1 - n, 0), t) : f_vertical_12(p, min(n, p.L2), t)

"BdG MPO of exchange_12 at time t on the existing `sites` (cutoffs as ramp_mpo)."
exchange12_mpo(t, g::Substrate, p::Exchange12, sites; cutoff=1e-16, P...) =
    first(msh_substrate(g, tspins(f_braid_12(p, t), g); sites, cutoff, profile_cutoff=1e-20, P...))
