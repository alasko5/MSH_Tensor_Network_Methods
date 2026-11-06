# Static zero modes of a T-junction on the substrate with the zero-mode pipeline (mirror start, X̂ localization).
# usage (from msh_substrate/): julia --project=.. checks/tjunction_static.jl L sabs.npz   (sabs from checks/tjunction_sabs.py)
# Result 2026-11-05 (single T): χ(ψ) 27 at 64²–256², E_f 3.12370–3.12374e-9 (sparse 3.1237201e-9), ΔQ −4.72e-8 (sparse −4.66e-8),
# gap 0.458729 (sparse same); Majorana centres 15.6 left and 14.6 right of the array centre.
include(joinpath(@__DIR__, "..", "substrate_model.jl"))
using NPZ, Printf
BLAS.set_num_threads(1)
L = parse(Int, ARGS[1]); g = Substrate(L, L, 12)
d = npzread(ARGS[2]); exch = Dict((d["ixy"][k, 1], d["ixy"][k, 2]) => Tuple(d["S"][k, :]) for k in 1:size(d["ixy"], 1))
t0 = time(); HB, sites = msh_substrate(g, exch); tH = time() - t0
z = zero_mode_subspace(HB, sites; xq=xqubits(g))
t1 = time(); r = fermion_from_subspace(HB, sites, z.ψs, g); tf = time() - t1
@printf("TN %d²: χ(H) %d | χ(ψ1, ψ2) %d %d | start2 %s | t_H %.1f t_K %.1f t_dmrg %.1f t_f %.1f s | Majorana x %.2f %.2f | E_f %.10e | ΔQ %+.10e | gap %.6f\n",
        2^L, maxlinkdim(HB), maxlinkdim.(z.ψs)..., z.start2[1], tH, z.t_K, z.t_dmrg, tf, r.centers..., r.E_f, ΔQ_mode(r.ψf), z.E_gap)
