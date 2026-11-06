# Static scaling of the single T-junction on the substrate, TN side: zero-mode pipeline (zero_mode_subspace + fermion_from_subspace,
# mirror start) as zm_run / cand_run of msh_substrate_tn.ipynb, single BLAS thread. One size per process (after a 64² warm-up that is
# not recorded), so that the peak RSS is that of the size; appends one row to tjunction_tn_scaling.csv.
# usage (from msh_substrate/): julia --project=.. checks/tjunction_scaling_tn.jl <bench|leak> L
# Result 2026-11-06 (benchmark point, each size alone): 61 s / 2.0 GB at 64² … 260 s / 2.6 GB at 2048², χ(ψ) 27 (32 at 2048²), χ(H) 49–52.
include(joinpath(@__DIR__, "..", "tjunction_substrate.jl"))
using Printf
BLAS.set_num_threads(1)
const OUT = "tjunction_tn_scaling.csv"
const COLS = ["point", "L", "N", "D", "t_H", "t_K", "t_dmrg", "t_pair", "t_total", "chi_H", "chi_K", "chi_psi", "mem_K_MB",
              "mem_psi_MB", "rss_MB", "E_f", "dQ", "E_gap", "E_sub1", "E_sub2", "x1", "x2", "start2"]
fT(c, s, n) = s == 3 ? 1.0 : 0.0                                   # horizontal arms topological, up arm trivial

function run(pt, L)
    g = Substrate(L, L, 12); P = pt == "bench" ? (;) : PC
    t_H = @elapsed ((HB, sites) = msh_substrate(g, tspins(fT, g); P...))
    z = zero_mode_subspace(HB, sites; xq=xqubits(g))
    t_pair = @elapsed (r = fermion_from_subspace(HB, sites, z.ψs, g))
    return Any[pt, L, nsites(g), 4nsites(g), t_H, z.t_K, z.t_dmrg, t_pair, t_H + z.t_K + z.t_dmrg + t_pair, maxlinkdim(HB),
               maxlinkdim(z.K), maximum(maxlinkdim, z.ψs), 16e-6 * mpo_size(z.K), 16e-6 * sum(mps_size, z.ψs), Sys.maxrss() / 1e6,
               r.E_f, ΔQ_mode(r.ψf), z.E_gap, z.E_sub..., r.centers..., z.start2[1]]
end

pt, L = ARGS[1], parse(Int, ARGS[2])
run(pt, 6)                                                          # warm-up (compilation)
row = run(pt, L)
isfile(OUT) || open(io -> println(io, join(COLS, ",")), OUT, "w")
open(io -> println(io, join(row, ",")), OUT, "a")
@printf("TN %s %d²: total %.0f s (H %.0f, K %.0f, DMRG %.0f) | χ H %d K %d ψ %d | RSS %.0f MB | E_f %.6e ΔQ %+.4e gap %.6f | ‖Hψ‖ %.1e %.1e | %s\n",
        pt, 2^L, row[9], row[5], row[6], row[7], row[10], row[11], row[12], row[15], row[16], row[17], row[18], row[19], row[20], row[23])
