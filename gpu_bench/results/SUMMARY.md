# GPU benchmark: sparse step and TDVP step (2026-11-04)

Measured on the GPU workstation, following `gpu_bench/README.md`. All numbers are seconds per time step (δt = 0.1, mid-ramp),
complex128 throughout, median of the repetitions after a first (warm-up) one, as in the reference measurements.

## Hardware and software

| | |
|---|---|
| host | morrtex24b (shared workstation, Linux 5.14.21, openSUSE) |
| CPU | Intel Core i9-14900K, 24 cores (8 P + 16 E), 32 threads; AVX2/FMA, **no AVX-512**. Single-thread runs pinned to P-core CPU 4 (`taskset -c 4`) |
| RAM | 188 GB |
| GPU | NVIDIA GeForce RTX 4070, 12 GB (consumer Ada card: FP64 throughput is 1/64 of FP32) |
| driver / CUDA | 570.86.16 / CUDA 12.8 (nvcc 12.8.61) |
| Python | conda env `~/.conda/envs/gpu_bench`: Python 3.12.16, NumPy 2.5.4, SciPy 1.18.1, CuPy 13.4.1 (CUDA runtime 12.8) |
| Julia | 1.13.2, env `gpu_bench/jlenv` (copy of the repo's Project/Manifest minus IJulia/Plots, plus CUDA.jl 6.5.1): TensorBinding 0.2.0, ITensors 0.9.32, ITensorMPS 0.3.44, OpenBLAS (Haswell kernel, ~83 GFLOP/s zgemm single thread) |

Load: the machine is shared (other users' Jupyter kernels); the load average was 0.7–2.1 during the runs (logged in
`gpu_bench/logs/*.log`), i.e. 1–2 busy cores out of 24. The single-thread CPU sparse step was run twice (before and after the
Julia precompilation load); the two runs agree to < 1 %.

## Results

Reference = the other machine (single thread, idle), from the README.

### Sparse step (benchmark point, ℓ = 12)

| | 512² | 1024² | 2048² | 4096² |
|---|---|---|---|---|
| reference CPU, 1 thread | 0.77 | 3.67 | 15.0 | |
| this CPU, 1 thread (total) | **0.70** | **3.36** | **13.5** | |
| ↳ H build / expm_multiply | 0.092 / 0.608 | 0.378 / 2.98 | 1.53 / 12.0 | |
| this CPU, all threads (total) | 0.69 | 3.32 | 13.3 | |
| GPU (total) | **0.196** | **0.768** | **3.15** | does not fit |
| ↳ H build (CPU) / transfer / exp (GPU) | 0.145 / 0.019 / 0.032 | 0.561 / 0.076 / 0.131 | 2.34 / 0.31 / 0.51 | |
| GPU memory in use | 0.9 GB | 2.7 GB | 9.7 GB | ~38 GB needed |

States: `psiT` from `td_py_psi_bench_{9_9,10_10}_rot10.npz`; 2048² random normalized vector at rot_T = 100 (as in the reference
`IDLE_CASES`, which has no 2048² rot 10 case). nnz(H) = 20.9 M / 83.8 M / 335 M.

- This CPU is ~9 % faster than the reference machine for the sparse step.
- GPU exponential alone: **19–24× faster** than single-thread CPU `expm_multiply`. Whole step: **3.6–4.4× faster**, because
  building H(t) on the CPU (and the −iδt scaling / trace for the GPU path, which is why the GPU-path H build is ~1.5× the CPU one)
  plus the PCIe transfer now take ~85 % of the GPU step.
- All threads give nothing: SciPy's sparse matrix–vector product is single-threaded.
- 4096² (L = 12) was not attempted: H alone is ~4× the 2048² footprint (~38 GB with the vectors and the CuPy pool) vs 12 GB.
- Not measured, but the obvious next step: H(t) differs from the time-independent part only on the 12 chain sites, so keeping
  `H_base` on the GPU and updating those diagonal blocks in place would remove the H build and transfer (≈ 2.6 s of the 3.15 s at
  2048²), leaving ≈ 0.5 s/step.

### TDVP step (two-site, `TB.evolve_with_tdvp_timedep`, krylovdim 30, tol 1e-12, reverse_step)

| | 512² (χ 74) | 1024² (χ 74) | 2048² (χ 75) | 2048² slow (χ 104) | leak 256², cap 128 | leak 256², cap 256 |
|---|---|---|---|---|---|---|
| reference CPU, 1 thread | 0.86 | 0.76 | 1.32 | | 5.0 | 14.9 |
| this CPU, 1 thread: reference call (H inside) | **0.90** | **0.76** | **1.32** | **1.33** | **9.42** | **29.1** |
| ↳ H build / sweep | 0.031 / 0.83 | 0.036 / 0.73 | 0.042 / 1.28 | 0.043 / 1.28 | 0.029 / 9.37 | 0.025 / 28.9 |
| GPU (total) | **2.15** | **2.22** | **2.43** | **2.52** | **4.06** | **8.68** |
| ↳ H build (CPU) / transfer / sweep (GPU) | 0.047 / 0.003 / 1.89 | 0.036 / 0.0002 / 2.19 | 0.114 / 0.004 / 2.31 | 0.121 / 0.004 / 2.40 | 0.029 / 0.0002 / 4.03 | 0.029 / 0.0002 / 8.65 |
| GPU / this CPU (total) | 2.4× slower | 2.9× slower | 1.8× slower | 1.9× slower | **2.3× faster** | **3.3× faster** |

- At the benchmark point (χ ≈ 74–104) the GPU is **slower**: the tensors are small, so the sweep is dominated by kernel
  launches, small SVDs/eigensolves and host–device synchronization, not by FLOPs, and the per-step cost is a nearly flat ~2.2–2.4 s.
- At the leaking point (χ = 128, 256) the GPU is **2.3× and 3.3× faster** than this CPU, and the gain grows with χ. In absolute
  terms the GPU is only 1.2× (cap 128) and 1.7× (cap 256) faster than the *reference* CPU, see the next point.
- **This CPU is ~2× slower than the reference machine at the leaking point** (9.4 vs 5.0 s, 29.1 vs 14.9 s), while it matches it
  at the benchmark point (0.90/0.76/1.32 vs 0.86/0.76/1.32). The leaking-point steps are dominated by dense zgemm, which runs ~2×
  faster with AVX-512, and this CPU has none (OpenBLAS uses its Haswell/AVX2 kernel). That is the most likely explanation but
  it is **not verified**: compare with `lscpu | grep -o avx512f` on the reference machine.
- The H(t) MPO build and the transfer are negligible for TDVP (≤ 0.12 s).

## Accuracy checks

- **Sparse:** one step at L = 9 (state `psiT`, t = T/2 + δt/2), GPU Taylor vs SciPy `expm_multiply` on the CPU:
  ‖ψ_gpu − ψ_cpu‖ = **1.3e-16** (step size ‖ψ_cpu − ψ₀‖ = 0.32), ‖ψ_gpu‖ − 1 = −4e-15.
- **TDVP:** for every state, one GPU step vs one CPU step from the same state with the same H(t):

  | state | \|⟨ψ_cpu\|ψ_gpu⟩\| − 1 | ‖ψ_cpu − ψ_gpu‖ | χ after (CPU / GPU) |
  |---|---|---|---|
  | bench 512² | −2.9e-15 | 3.7e-15 | 74 / 74 |
  | bench 1024² | −3.1e-15 | 3.4e-15 | 74 / 74 |
  | bench 2048² | 6.7e-16 | 4.4e-15 | 76 / 76 |
  | bench 2048² slow | −3.1e-15 | 8.7e-15 | 104 / 104 |
  | leak cap 128 | −4.4e-16 | 2.7e-15 | 128 / 128 |
  | leak cap 256 | −1.7e-15 | 3.3e-15 | 256 / 256 |

## Methods and caveats

- **Sparse GPU exponential** (`sparse_bench.py`, `gpu_expm_multiply`): the algorithm of SciPy's `expm_multiply` (Al-Mohy & Higham
  2011) ported to CuPy: trace shift, (m*, s) from the exact 1-norm of A = −iδtH with SciPy's θ_m table (tol 2⁻⁵³), truncated Taylor
  series with SciPy's early-termination test. Here m* = 26, s = 1 at every size. The matrix–vector products run in cuSPARSE via
  `cupyx.scipy.sparse.csr_matrix`. The step evolves ψ, as in `idle_step`.
- **H build** for the GPU sparse path includes forming A = −iδtH and its trace on the CPU. For the CPU path the scaling happens
  inside the `expm_multiply` call and is counted in "expm".
- **TDVP GPU:** the reference function `TB.evolve_with_tdvp_timedep` worked unchanged on GPU tensors. The MPS and the midpoint
  MPO are moved with `adapt(CuArray, ·)` (ComplexF64 kept, asserted), and the function is given `_ -> H_gpu`. It calls `Hoft` once
  at t = δt/2, so this is the same computation as the reference call with the H build taken out. The TDVP step always starts from
  the stored state (as in the reference cell). The CPU "reference_call" row is the reference cell unchanged. The CPU "total" is H
  build + sweep timed separately, and the two agree.
- **GPU memory** for TDVP is device-wide use from `nvidia-smi`, so it includes the X server (~140 MB) and CUDA.jl's cached pool. It
  is an upper bound (2.9–7.2 GB).
- **Single thread:** `OMP/MKL/OPENBLAS_NUM_THREADS=1`, `BLAS.set_num_threads(1)`, one Julia thread, pinned to one P-core. The
  reference machine was probably not pinned. FTZ/DAZ was not used (the brief says to leave it off).
- The GPU H-build numbers for TDVP at 2048² (0.11–0.12 s vs 0.04 s on the CPU run) are probably CUDA.jl's host threads sharing
  the pinned core. Either way, the effect is small.

## Files

- `sparse_bench.py`: sparse CPU/GPU step and accuracy check. `run_sparse.sh` (CPU 1 thread + GPU, first run),
  `run_sparse_cpu.sh` (CPU 1 thread repeat + all threads, the CPU rows in the CSV).
- `tdvp_bench.jl`, `run_tdvp.sh`: TDVP CPU/GPU step and accuracy check.
- `timings.csv`: `method, device, point, L, rot_T, chi, part, seconds`. `part` is H_build, transfer, step (the exponential/sweep
  only), total, plus `reference_call` for the unmodified CPU TDVP call. `device` is `cpu_1thr`, `cpu_32thr` or `gpu`.
- Environment setup: `gpu_bench/jlenv/setup.jl`. Logs: `gpu_bench/logs/`.

## Note added by the CPU-machine session (2026-11-04)

The AVX-512 explanation above is ruled out: the reference machine (morrtex24a) has the same CPU, an i9-14900K without AVX-512.
The leaking-point reference times in the README (5.0 s and 14.9 s) were measured with the **old** Hamiltonian MPO cutoff
(1e-13). The benchmark-point reference times were re-measured with the current code (`ramp_mpo` at 1e-16), which is also what this
benchmark used. So the 2× at the leaking point is most likely the cost of the larger H(t) MPO at χ = 128–256, not the hardware. The
idle re-timing on the reference machine, after the current reruns, will check this.
