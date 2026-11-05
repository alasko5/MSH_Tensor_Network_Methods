# E_f at the benchmark point: MPO expectation <ψf|H|ψf> of the TN zero mode, and the dense vector of the same ψf in the Python
# usage (from msh_substrate/): julia --project=.. checks/ef_mpo_vs_sparse.jl L, then python checks/ef_mpo_vs_sparse.py L
# Result 2026-11-05 (benchmark): 2048²: <ψ|H_sparse|ψ> = 3.71228475084e-4 (exact 3.7122847509e-4), <ψ|H_MPO|ψ> = 3.71228468934e-4.
# basis (data/efcheck/psi_<L>.npy) for the sparse-H expectation in ef_check.py.  args: L
include(joinpath(@__DIR__, "..", "substrate_model.jl"))
using NPZ, Printf
BLAS.set_num_threads(1)
L = parse(Int, ARGS[1]); g = Substrate(L, L, 12); N = Nx(g) * Ny(g)
HB, sites = msh_substrate(g, S_z)
z = zero_mode_subspace(HB, sites; xq = xqubits(g)); r = fermion_from_subspace(HB, sites, z.ψs, g)
ψ = r.ψf / norm(r.ψf)
E_mpo = real(inner(ψ', HB, ψ))
v = vec(Array(ITensors.@disable_warn_order(prod(ψ)), reverse(sites)...))
w = zeros(ComplexF64, 4N)
for m in 0:N-1
    w[4m+1] = v[m+1]; w[4m+2] = v[N+m+1]; w[4m+3] = v[3N+m+1]; w[4m+4] = -v[2N+m+1]
end
npzwrite("data/efcheck/psi_$(L).npy", w)
@printf("L %d: E_f (pipeline) %.13e  <ψ|H_MPO|ψ> %.13e  ΔQ %+.12e  χ(H) %d\n", L, r.E_f, E_mpo, ΔQ_mode(ψ), maxlinkdim(HB))
