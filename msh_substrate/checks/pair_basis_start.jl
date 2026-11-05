# DMRG pair basis test: random start (current pipeline) against a start localized on the left half (MSB x qubit = 0).
# usage (from msh_substrate/): julia --project=.. checks/pair_basis_start.jl <bench|leak> L <random|left>
# Result 2026-11-05 (leak, 512²): random start χ 137/137, centres 249.5/261.5; left start χ 114/114, centres 285.9/225.1; same E_f, ΔQ to 1e-9.
# args: point L mode   (mode = "random" | "left")
include(joinpath(@__DIR__, "..", "substrate_model.jl"))
using Printf
BLAS.set_num_threads(1)
pt, L, mode = ARGS[1], parse(Int, ARGS[2]), ARGS[3]
P = pt == "bench" ? (;) : PC; ℓ = pt == "bench" ? 12 : 64
g = Substrate(L, L, ℓ); HB, sites = msh_substrate(g, S_z; P...); xq = xqubits(g)
"zero_mode_subspace with an optional first start; same settings as the pipeline."
function zms(HB, sites; start1=nothing, nzero=2, η=1e-2, seed=1234, nsweeps=20, maxdim=[10, 20, 50, 100, 200], cutoff=1e-14,
             cutoff_K=1e-20, noise=[1e-6, 1e-7, 1e-8, 0.0], weight=10.0, eigsolve_krylovdim=20, eigsolve_maxiter=3)
    t_K = @elapsed (K = TB.build_K(HB, sites, 0.0, η; cutoff_K))
    Random.seed!(seed); ψs = MPS[]; chis = Int[]; ts = Float64[]
    orth(φ) = +(φ, -inner(ψs[1], φ) * ψs[1]; cutoff=1e-14)
    for k in 1:nzero+1
        ψ0 = random_mps(sites; linkdims=10)
        if k == 1 && start1 !== nothing
            ψ0 = start1(ψ0)
        elseif k == 2
            m = orth(mirror_x(ψs[1], sites, xq)); ψ0 = m / norm(m)
        end
        t0 = time()
        _, ψ = quietly() do
            isempty(ψs) ? dmrg(K, ψ0; nsweeps, maxdim, cutoff, noise, eigsolve_krylovdim, eigsolve_maxiter, outputlevel=0) :
                          dmrg(K, ψs, ψ0; weight, nsweeps, maxdim, cutoff, noise, eigsolve_krylovdim, eigsolve_maxiter, outputlevel=0)
        end
        t = time() - t0
        push!(ψs, ψ / norm(ψ)); push!(chis, maxlinkdim(ψ)); push!(ts, t)
    end
    ψgap = pop!(ψs)
    return (; ψs, E_gap = norm_Hψ(HB, ψgap), K, t_K, chis, ts)
end
left(ψ) = (j = first(xq); φ = copy(ψ); φ[j] = noprime(ITensor([1.0 0.0; 0.0 0.0], sites[j]', sites[j]) * φ[j]); φ / norm(φ))
z = zms(HB, sites; start1 = mode == "left" ? left : nothing)
r = fermion_from_subspace(HB, sites, z.ψs, g)
# x centre of each DMRG state (how localized the returned basis is)
n1(j) = ITensor([0.0 0.0; 0.0 1.0], sites[j]', sites[j])
xc(ψ) = real(sum(2.0^(g.Lx - k) * inner(ψ, apply_site(n1(j), ψ, j)) for (k, j) in enumerate(xq)))
@printf("%s %d² %s: χ(ψ1, ψ2, gap) = %s | DMRG s (ψ1, ψ2, gap) = %s, total %.0f | x centre ψ1 %.1f ψ2 %.1f | Majoranas %.1f %.1f | E_f %.10e | ΔQ %+.8e | gap %.5f\n",
        pt, 2^L, mode, z.chis, round.(Int, z.ts), sum(z.ts), xc(z.ψs[1]), xc(z.ψs[2]), r.centers..., r.E_f, ΔQ_mode(r.ψf), z.E_gap)
