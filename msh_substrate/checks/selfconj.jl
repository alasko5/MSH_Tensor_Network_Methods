# Self-conjugacy check of the localized Majoranas of the zero-mode pipeline (fermion_from_subspace).
# Per case: |<γ|Cγ>| for each localized Majorana before the phase fix (1 − it = residual), ‖Cγ − γ‖ after,
# and how C-invariant the DMRG subspace is (weight of Cψ_k outside span{ψ1, ψ2}).
#
# usage (from msh_substrate/): julia --project=.. checks/selfconj.jl <bench|leak> Lx Ly
# Results of 2026-11-05 (bench 32²–2048², leak 128×64–512²): subspace C-leak ≤ 1e-12, 1 − |<γ|Cγ>| ≤ 6e-14,
# ‖Cγ − γ‖ ≤ 4e-7 (= sqrt of 2(1 − Re<γ|Cγ>)), |<ψf|Cψf>| ≤ 2e-9.
include(joinpath(@__DIR__, "..", "substrate_model.jl"))
using Printf
BLAS.set_num_threads(1)
cases = [(ARGS[1], parse(Int, ARGS[2]), parse(Int, ARGS[3]))]
for (pt, Lx, Ly) in cases
    P = pt == "bench" ? (;) : PC; ℓ = pt == "bench" ? 12 : 64
    g = Substrate(Lx, Ly, ℓ)
    HB, sites = msh_substrate(g, S_z; P...)
    z = zero_mode_subspace(HB, sites; xq = xqubits(g))
    r = fermion_from_subspace(HB, sites, z.ψs, g)
    C = r.C
    # subspace C-invariance: 1 − Σ_j |<ψ_j|Cψ_k>|² for k = 1, 2
    leak = [1 - sum(abs2(inner(ψj, C(ψk))) for ψj in z.ψs) for ψk in z.ψs]
    # residual of each final γ (after the phase fix): 1 − |<γ|Cγ>| and ‖Cγ − γ‖
    res = [1 - abs(inner(γ, C(γ))) for γ in r.γs]
    dist = [norm(+(C(γ), -γ; cutoff=1e-16)) for γ in r.γs]
    ov = abs(inner(r.ψf, C(r.ψf)))              # <ψf|Cψf> should be 0 (ψf ⟂ its conjugate)
    @printf("%s %d×%d: E_f %.6e | subspace C-leak %.1e %.1e | 1−|<γ|Cγ>| %.1e %.1e | ‖Cγ−γ‖ %.1e %.1e | |<ψf|Cψf>| %.1e | centres %.1f %.1f\n",
            pt, 2^Lx, 2^Ly, r.E_f, leak..., res..., dist..., ov, r.centers...)
end
