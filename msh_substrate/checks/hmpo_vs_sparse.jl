# MPO H(t) against the sparse H(t) on a ramp state: writes the dense ψ and H_MPO(t)ψ (Python basis) for hcheck.py.  args: L
# usage (from msh_substrate/): julia --project=.. checks/hmpo_vs_sparse.jl L, then python checks/hmpo_vs_sparse.py L (fast benchmark ramp, final state)
# Result 2026-11-05: ‖(H_MPO(t) − H_sparse(t))ψ‖ = 0.7–15e-9 (2048²), 2.5–8e-9 (4096²) at t = 0, 3, 6, 9, 13.
include(joinpath(@__DIR__, "..", "substrate_model.jl"))
using NPZ, Printf, Serialization
BLAS.set_num_threads(1)
L = parse(Int, ARGS[1]); g = Substrate(L, L, 12); N = Nx(g) * Ny(g); rot = 10.0
d = deserialize("data/td_tn_large_bench_$(L)_$(L)_rot10_m256_c1.0e-14.jls"); ψ = d.ψ / norm(d.ψ); sites = d.sites
function pyv(φ)
    v = vec(Array(ITensors.@disable_warn_order(prod(φ)), reverse(sites)...))
    w = zeros(ComplexF64, 4N)
    for m in 0:N-1
        w[4m+1] = v[m+1]; w[4m+2] = v[N+m+1]; w[4m+3] = v[3N+m+1]; w[4m+4] = -v[2N+m+1]
    end
    return w
end
npzwrite("data/efcheck/h_psi_$(L).npy", pyv(ψ))
for t in (0.0, 3.0, 6.0, 9.0, 13.0)
    H = ramp_mpo(t, g, rot, sites)
    e = real(inner(ψ', H, ψ))
    Hψ = apply(H, ψ; alg="naive", truncate=false)
    npzwrite("data/efcheck/h_Hpsi_$(L)_t$(round(Int, t)).npy", pyv(Hψ))
    @printf("L %d t %4.1f: <ψ|H_MPO|ψ> %.15e  χ(H) %d  χ(Hψ) %d\n", L, t, e, maxlinkdim(H), maxlinkdim(Hψ)); flush(stdout)
end
