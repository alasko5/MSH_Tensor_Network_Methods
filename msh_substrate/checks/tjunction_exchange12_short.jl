# exchange_12 on one T-junction, short stretch: TDVP of ψ_f from the static zero mode over [t0, t1], snapshots every 10 time units,
# for checks/tjunction_exchange12_short.py (sparse reference from the same ψ(t0)). The texture moves from t = 0 (arm 1 retracts over
# t = 0 … 320), so the static state is the right start only for t0 = 0; a later t0 is a quench.
# usage (from msh_substrate/): julia --project=.. checks/tjunction_exchange12_short.jl L t0 t1
# Result 2026-11-06 (256², t0 = 0 → 50): χ 35 → 63, 0.45–0.71 s/step, 1 − F vs sparse ≤ 2.1e-10, ⟨H⟩ = E_f = 3.1e-9 throughout.
include(joinpath(@__DIR__, "..", "tjunction_substrate.jl"))
using NPZ, Printf
BLAS.set_num_threads(1)
L, t0, t1 = parse(Int, ARGS[1]), parse(Float64, ARGS[2]), parse(Float64, ARGS[3]); DT = 0.1
g = Substrate(L, L, 12); p = Exchange12(); N = Nx(g) * Ny(g); out = "data/tjunction/ex12_short_$(L)"
HB, sites = msh_substrate(g, tspins(f_braid_12(p, 0.0), g))
z = zero_mode_subspace(HB, sites; xq=xqubits(g)); r = fermion_from_subspace(HB, sites, z.ψs, g)
@printf("static: E_f %.6e ΔQ %+.3e χ %d\n", r.E_f, ΔQ_mode(r.ψf), maxlinkdim(r.ψf))
function pyv(φ)
    v = vec(Array(ITensors.@disable_warn_order(prod(φ)), reverse(sites)...)); w = zeros(ComplexF64, 4N)
    for m in 0:N-1; w[4m+1] = v[m+1]; w[4m+2] = v[N+m+1]; w[4m+3] = v[3N+m+1]; w[4m+4] = -v[2N+m+1]; end
    return w
end
tex = tspins(f_braid_12(p, 345.0), g); k = sort(collect(keys(tex)))
npzwrite(out * "_tex345.npz", Dict("ixy" => [a[i] for a in k, i in 1:2], "S" => [tex[a][i] for a in k, i in 1:3]))
ψ = r.ψf; npzwrite(out * "_t$(round(Int, t0)).npy", pyv(ψ))
tb = @elapsed exchange12_mpo(t0, g, p, sites); @printf("H(t) build %.2f s, χ(H) %d\n", tb, maxlinkdim(exchange12_mpo(t0, g, p, sites)))
nsnap = round(Int, 10 / DT); t = t0
while t < t1 - 1e-9
    ts = time()
    global ψ = TB.evolve_with_tdvp_timedep(τ -> exchange12_mpo(t + τ, g, p, sites), ψ, nsnap, DT; maxdim=256, cutoff=1e-14,
                                           tol=1e-12, krylovdim=30, reverse_step=true)[end]
    global t += nsnap * DT
    npzwrite(out * "_t$(round(Int, t)).npy", pyv(ψ))
    @printf("t %.0f: χ %d, %.2f s/step, ΔQ %+.6e, ⟨H⟩ %+.6e\n", t, maxlinkdim(ψ), (time() - ts) / nsnap, ΔQ_mode(ψ),
            real(inner(ψ', exchange12_mpo(t, g, p, sites), ψ))); flush(stdout)
end
