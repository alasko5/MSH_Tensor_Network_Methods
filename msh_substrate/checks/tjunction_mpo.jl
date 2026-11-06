# Hamiltonian MPO with exchange on arbitrary sites (T-junction) against the sparse H: writes ψ and H_MPO ψ (Python basis).
# usage (from msh_substrate/): python checks/tjunction_sabs.py 6 1 data/tjunction/sabs_6_1.npz;
#   julia --project=.. checks/tjunction_mpo.jl 6 data/tjunction/sabs_6_1.npz; python checks/tjunction_mpo.py 6 data/tjunction/sabs_6_1.npz
# Result 2026-11-05 (single T, 64²): chain row as Dict = function form exactly (χ 40); T MPO χ 51, ‖(H_MPO − H_sparse)ψ‖ = 8.7e-14.
include(joinpath(@__DIR__, "..", "substrate_model.jl"))
using NPZ, Printf
BLAS.set_num_threads(1)
L = parse(Int, ARGS[1]); g = Substrate(L, L, 12); N = Nx(g) * Ny(g)
H1, sites = msh_substrate(g, S_tilt)
H2, _ = msh_substrate(g, Dict((ix, chain_row(g)) => S_tilt(ix) for ix in chain_x(g)); sites)
@printf("chain: function vs Dict ‖ΔH‖/‖H‖ %.2e, χ %d %d\n", norm(+(H1, -1.0 * H2; cutoff=1e-30)) / norm(H1), maxlinkdim(H1), maxlinkdim(H2))
d = npzread(ARGS[2]); exch = Dict((d["ixy"][k, 1], d["ixy"][k, 2]) => Tuple(d["S"][k, :]) for k in 1:size(d["ixy"], 1))
HT = first(msh_substrate(g, exch; sites))
@printf("T: %d magnetic sites on %d rows, χ(H) %d (chain %d)\n", length(exch), length(unique(last.(keys(exch)))), maxlinkdim(HT), maxlinkdim(H1))
Random.seed!(7); ψ = random_mps(sites; linkdims=4); ψ /= norm(ψ)
function pyv(φ)
    v = vec(Array(ITensors.@disable_warn_order(prod(φ)), reverse(sites)...)); w = zeros(ComplexF64, 4N)
    for m in 0:N-1; w[4m+1] = v[m+1]; w[4m+2] = v[N+m+1]; w[4m+3] = v[3N+m+1]; w[4m+4] = -v[2N+m+1]; end
    return w
end
npzwrite("data/tjunction/mpo_psi_$L.npy", pyv(ψ)); npzwrite("data/tjunction/mpo_Hpsi_$L.npy", pyv(apply(HT, ψ; alg="naive", truncate=false)))
