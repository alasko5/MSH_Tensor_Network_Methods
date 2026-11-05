# Zero-bias LDOS cuts from the TN zero mode (as zerobias_tn in msh_substrate_ldos_tn.ipynb, section 2), with the fixed MPO,
# plus the infidelity 1 − |<ψ_TN|ψ_sparse>|² against the sparse ψ_f saved by the time-dependent runs (psi0 of td_py_psi_*).
# usage (from msh_substrate/): julia --project=.. checks/zerobias_overlap.jl <bench|leak> L   (L = 8 … 11)
# Results of 2026-11-05: 1 − F = 6–10e-14 (benchmark), 2–4e-13 (leaking) at 256²–1024²; no sparse ψ_f at 2048².
#  Writes data/ldos/zb_tn_<pt>_<L>_<L>.npz and data/ldos/zb_rerun_<pt>_<L>.csv (one row).
include(joinpath(@__DIR__, "..", "substrate_model.jl"))
using NPZ, Printf
BLAS.set_num_threads(1)
const DATA = "data/ldos"; const H_WIN = 64
const POINTS = Dict("bench" => (P = NamedTuple(), ℓ = 12), "leak" => (P = PC, ℓ = 64))
function py_vector(ψ, sites, N)
    v = vec(Array(ITensors.@disable_warn_order(prod(ψ)), reverse(sites)...))
    w = zeros(ComplexF64, 4N)
    for m in 0:N-1
        w[4m+1] = v[m+1]; w[4m+2] = v[N+m+1]; w[4m+3] = v[3N+m+1]; w[4m+4] = -v[2N+m+1]
    end
    return w
end
pt, L = ARGS[1], parse(Int, ARGS[2])
P, ℓ = POINTS[pt]; g = Substrate(L, L, ℓ); nx = Nx(g); x0, iy0 = first(chain_x(g)), chain_row(g)
t_dmrg = @elapsed begin
    H0, sites = msh_substrate(g, S_z; P...)
    z = zero_mode_subspace(H0, sites; xq = xqubits(g)); r = fermion_from_subspace(H0, sites, z.ψs, g)
end
ψ = r.ψf / norm(r.ψf)
cols = max(0, x0 - H_WIN):min(nx - 1, x0 + ℓ + H_WIN - 1); rows = max(0, iy0 - H_WIN):iy0 + H_WIN - 1
t_cut = @elapsed begin
    row = 2 .* vec(sum(row_weights(ψ, sites, g, iy0, 0:nx-1); dims=1))
    cross_end = 2 .* vec(sum(col_weights(ψ, sites, g, x0, 0:nx-1); dims=1))
    win = 2 .* vcat([sum(row_weights(ψ, sites, g, iy, cols); dims=1) for iy in rows]...)
end
rss = Sys.maxrss() / 2^20
npzwrite(joinpath(DATA, "zb_tn_$(pt)_$(L)_$(L).npz"), Dict("row" => row, "cross_end" => cross_end, "win" => win, "x0" => [x0], "iy0" => [iy0], "ell" => [ℓ]))
f = "data/td_py_psi_$(pt)_$(L)_$(L)_rot100.npz"
inf = NaN
if isfile(f)
    ps = npzread(f)["psi0"]; w = py_vector(ψ, sites, nx * nx)
    inf = 1 - abs2(dot(ps, w)) / (norm(ps)^2 * norm(w)^2)
end
open(joinpath(DATA, "zb_rerun_$(pt)_$(L).csv"), "w") do io
    println(io, join(Any[pt, nx, nx, t_dmrg, t_cut, rss, maxlinkdim(ψ), r.E_f, ΔQ_mode(ψ), inf], ","))
end
@printf("%s %d²: DMRG %.0f s, cuts %.0f s, RSS %.0f MB, χ %d, E_f %.10e, 1 − F %.2e\n", pt, nx, t_dmrg, t_cut, rss, maxlinkdim(ψ), r.E_f, inf)
