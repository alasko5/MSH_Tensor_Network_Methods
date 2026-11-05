# GPU benchmark brief: sparse step and TDVP step on the GPU workstation

Written 2026-11-04 by the Claude Code session running the MSH-substrate study on the CPU machine. You (a Claude Code instance on
the GPU workstation) share the home directory with that machine. Read this whole file before running anything.

## Goal

The study compares tensor networks (quantics MPS/MPO, two-site TDVP) with a sparse reference (`scipy.sparse.linalg.expm_multiply`)
for a spin ramp of a magnetic chain on a 2D superconducting substrate, 2^L × 2^L sites. All timings so far are **single CPU thread,
one process, idle machine**. Question: how does the cost per time step change on a GPU, for each method? Measure; do not estimate.

Deliverable: `gpu_bench/results/` with the scripts you ran, a CSV of timings, and `SUMMARY.md` (hardware, software versions, table,
caveats). Nothing else.

## Hard rules (shared home directory, jobs running on the other machine)

- **Read-only** everywhere except `~/MSH_Tensor_Network_Methods/gpu_bench/`. In particular do **not** modify, move or delete anything
  in `~/MSH_Tensor_Network_Methods/msh_substrate/` (its `data/` is being written by running jobs), and do not run `git` commands that
  change the working tree (no checkout, switch, stash, reset, commit, pull). The branch `mpo-cutoff-fix` has uncommitted work.
- **Julia:** do not add packages to the repository's environment (`~/MSH_Tensor_Network_Methods/Project.toml`) or to the default
  `~/.julia/environments/v1.13`. Make a separate environment in `gpu_bench/jlenv/` (`] activate gpu_bench/jlenv`, then add
  `TensorBinding` 0.2.0, `ITensors` 0.9.32, `ITensorMPS` 0.3.44 (the versions of the repo's `Manifest.toml`), `CUDA`, `NPZ`,
  `Serialization`; `substrate_model.jl` also needs the packages it `using`s — read its first lines).
  Never run `Pkg.gc()` or `Pkg.update()` on the shared depot `~/.julia`. Julia binary:
  `~/.julia/juliaup/julia-1.13.2+0.x64.linux.gnu/bin/julia`.
- **Python:** do not install into existing conda envs (`myenv`, `julia_env`, `claude_cli`, …). Create a new env
  `~/.conda/envs/gpu_bench` with `numpy scipy cupy` (matching the workstation's CUDA version).
- Ask the user before anything that needs root, or before running anything longer than about 30 minutes.

## Files to read (read-only)

- `msh_substrate/substrate_model.py`: `Ramp(Lx, Ly, ell, rot_T, **P)` gives the sparse BdG `H(t)` (`R.H(t)`, CSR, dimension
  4N, N = 4^L), `R.T` the ramp duration. Benchmark point: `P = {}` (μ = −7, Δ = 2.4, J = 5.8), ℓ = 12. Leaking point:
  `P = dict(mu=-3.0, Delta=0.6, J=3.0)`, ℓ = 64.
- `msh_substrate/msh_substrate_td_python.ipynb`, cell 17 (`idle_step`): **the CPU reference measurement**. One step =
  build `H(t)` at mid-ramp (t₀ = T/2, midpoint t₀ + (s + ½)δt, δt = 0.1) + `expm_multiply(-1j*δt*H, psi)`, median of 5 after a
  first one, FTZ/DAZ on (`ftz_daz(True)`, needs `msh_substrate/ftz.c` compiled; on the GPU side not needed).
- `msh_substrate/substrate_model.jl`: `Substrate`, `ramp_mpo(t, g, rot_T, sites; P...)`, `PC` (leaking-point parameters).
- `msh_substrate/msh_substrate_td_tn.ipynb`, cell 20 (`RUN_IDLE`): **the CPU TDVP step measurement**: load the final state, one
  `TB.evolve_with_tdvp_timedep(t -> ramp_mpo(tm + t, g, rot, d.sites; P...), d.ψ, 1, 0.1; maxdim, cutoff=1e-14, tol=1e-12,
  krylovdim=30, reverse_step=true)` at tm = T/2, median of 3 after a first one.

States (Julia `Serialization`, `(; ψ, sites)`), read-only:

| case | file | χ of the state |
|---|---|---|
| benchmark, fast ramp, 512² / 1024² / 2048² | `msh_substrate/data/mpo16/td_tn_large_bench_{9_9,10_10,11_11}_rot10_m256_c1.0e-14.jls` | ~74 |
| benchmark, slow ramp, 2048² | `msh_substrate/data/mpo16/td_tn_large_bench_11_11_rot100_m256_c1.0e-14.jls` | ~104 |
| leaking point, 256², cap 128 | `msh_substrate/data/td_tn_large_leak_8_8_rot100_m128_c1.0e-14.jls` | 128 |
| leaking point, 256², cap 256 | `msh_substrate/data/td_tn_large_leak_8_8_rot100_m256_c1.0e-12.jls` | 256 |

(`rot` = 10 for `rot10`, 100 for `rot100`; leaking point ℓ = 64, `P = PC`; benchmark ℓ = 12.) Sparse states for 512²/1024² are in
`msh_substrate/data/td_py_psi_bench_{9_9,10_10}_rot10.npz` (`psiT`); a random normalized vector is fine for timing.

CPU reference numbers from the other machine (single thread, idle), seconds per step:

| | 512² | 1024² | 2048² |
|---|---|---|---|
| sparse, benchmark | 0.77 | 3.67 | 15.0 |
| TDVP, fast benchmark ramp (χ ≈ 74) | 0.86 | 0.76 | 1.32 |
| TDVP, leaking point 256², cap 128 / 256 | 5.0 / 14.9 (256²) | | |

## What to measure

1. **Inventory**: `nvidia-smi`, CPU model and core count (`lscpu`), RAM, CUDA version. Put them in `SUMMARY.md`.
2. **Sparse step, this machine's CPU, single thread** (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`):
   reproduce `idle_step` (without FTZ) for the benchmark point at L = 9, 10, 11. This calibrates this CPU against the reference
   machine.
3. **Sparse step on the GPU**: build `H(t)` on the CPU with `Ramp.H` (time it separately), move it to the GPU
   (`cupyx.scipy.sparse.csr_matrix`, time the transfer separately) and apply exp(−iδtH)ψ on the GPU. CuPy has no `expm_multiply`;
   write a truncated Taylor series or a Krylov (Arnoldi/Lanczos) step with tolerance ~1e-12, and **check it against scipy's
   `expm_multiply` on the CPU** for one step at L = 9 (‖ψ_gpu − ψ_cpu‖ ≲ 1e-10). Report: H build, transfer, exponential, and the
   total per step, at L = 9, 10, 11 (and 12 if it fits in GPU memory; report the memory used).
4. **TDVP step, this machine's CPU, single thread**: reproduce the CPU TDVP step measurement for the states above
   (`BLAS.set_num_threads(1)`).
5. **TDVP step on the GPU**: the same step with the MPS and the MPO on the GPU (ITensors' CUDA extension: `using CUDA`, then
   `cu(ψ)`, `cu(H)`; `ramp_mpo` builds on the CPU, so move each H(t) to the GPU and time that). If
   `TB.evolve_with_tdvp_timedep` does not work with GPU tensors, use ITensorMPS's `tdvp` with the midpoint H for one step, and say
   which you used. Check that the GPU step gives the same state as the CPU step (overlap to ~1e-10). Report per state: H build,
   transfer, TDVP sweep, total.
6. Optionally, if quick: the CPU sparse step with all cores (threads unrestricted), for comparison with the GPU.

Precision: double (complex128) throughout, as on the CPU.

## Output

`gpu_bench/results/timings.csv` with columns `method, device, point, L, rot_T, chi, part, seconds` (part = H_build, transfer,
step, total), the scripts, and `SUMMARY.md`: hardware and versions, a table like the reference table above for this machine
(CPU single thread, GPU), the accuracy checks, and anything that did not work and why. Keep the logs of long outputs in files and
show only the relevant lines.
