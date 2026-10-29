# MSH Tensor Network Methods

Tensor-network (MPO/MPS) calculations for magnet–superconductor hybrid (MSH) systems and the Kitaev chain, built with
TensorBinding.jl on top of ITensors. Every tensor-network result is checked
against a dense reference: exact diagonalization or exact time evolution of the same model, in Julia or in the
Python/kwant notebooks.

## Notebooks

| Notebook | Language | What it does |
| --- | --- | --- |
| `msh_chain_tn.ipynb` | Julia | Single MSH chain as an MPO. LDOS by exact diagonalization and by TensorBinding's KPM at four snapshots of the spin ramp (fully topological → fully trivial), Majorana zero modes and the fermion mode (DMRG), charge difference ΔQ between the logical states and the time-dependent spin ramp (TDVP). Saves its results to `msh_chain_tn_results.npz`. |
| `msh_chain_python_ed.ipynb` | Python | Dense NumPy reference for `msh_chain_tn.ipynb`: same model, same ramp snapshots and protocols. Writes `msh_compare_python.npz`. |
| `msh_chain_compare.ipynb` | Python | Number-by-number comparison of the two MSH chain notebooks: loads `msh_chain_tn_results.npz` and `msh_compare_python.npz`, prints one ✓/✗ row per quantity (dense against dense to round-off, tensor networks against dense to their truncation error) and plots the overlays. |
| `msh_tjunctions_tn.ipynb` | Julia | Three-T-junction MSH array, with MPO/MPS methods only: MPO Hamiltonian (checked against the dense lattice matrix), subgap LDOS and Majoranas from DMRG, and the Z and X gates with partial-fusion readout (leakage of each mode, ΔQ, δn(x, t)) from TDVP. Ends with the cost of the tensor-network route and its scaling with system size. |
| `tjunctions_python_ed.ipynb` | Python | kwant/exact-diagonalization reference for `msh_tjunctions_tn.ipynb`: same geometry and protocols. Checks that kwant builds the same Hamiltonian as the TensorBinding MPO (`tb_reference/`), then the subgap LDOS, and the Z and X gates with partial fusion in the construction of the CZ, QFT and period-finding notebooks (`individual_MZMs`, `fix_unitary`, logical-state masks, full-basis Gauss–Legendre evolution): leakage, ΔQ and δn(x, t), and the cost of the dense route. Writes `tjunctions_python_ed_data/`. |
| `tjunctions_compare.ipynb` | Python | Number-by-number comparison of the two T-junction notebooks: LDOS, MPS/TDVP δn(x, t) and ΔQ(t), and the leakage and fusion quantities against the dense Python results (one ✓/✗ row each), plus the runtimes and the scaling of both methods with system size. |
| `kitaev_charge_difference.ipynb` | Julia | Kitaev chain: fermion mode from the DMRG Majorana doublet, KPM LDOS, McWeeny bulk projector with ϵF = −(E_mode + E_gap)/2, ΔQ versus topological length, and a time-dependent smoothstep ramp that fuses the Majoranas (TDVP, with TDVP and McWeeny bulk densities). |

## Documentation

Function references (PDF) in `docs/`. Each describes every function in its notebooks: method, inputs and outputs, numerical
settings, and the printed results and figures.

| File | Covers |
| --- | --- |
| `docs/kitaev_charge_difference_function_reference.pdf` | `kitaev_charge_difference.ipynb` |
| `docs/msh_chain_function_reference.pdf` | `msh_chain_tn.ipynb`, `msh_chain_python_ed.ipynb`, `msh_chain_compare.ipynb`, and the TensorBinding routines they use |
| `docs/msh_tjunctions_function_reference.pdf` | `msh_tjunctions_tn.ipynb`, `tjunctions_python_ed.ipynb`, `tjunctions_compare.ipynb` |

The LaTeX sources are in `docs/src/`; `docs/src/build.sh` rebuilds the PDFs from them and the notebook outputs (see
`docs/src/README.md`).

## Data files

Committed CSVs (small results that other notebooks read):

| File | Written by | Read by |
| --- | --- | --- |
| `msh_array_ldos_tn.csv` | `msh_tjunctions_tn.ipynb` | `tjunctions_compare.ipynb` |
| `msh_gate_density_tdvp_{Z,X}.csv`, `msh_gate_dQ_tdvp_{Z,X}.csv` | `msh_tjunctions_tn.ipynb` (TDVP, section 6) | `tjunctions_compare.ipynb` |
| `msh_tjunctions_tn_gates.csv`, `msh_tjunctions_tn_timing.csv` | `msh_tjunctions_tn.ipynb` (sections 8 and 9) | `tjunctions_compare.ipynb` |

Also committed, so that the two compare notebooks run from a fresh clone without rerunning anything:

| File | Written by | Read by |
| --- | --- | --- |
| `msh_chain_tn_results.npz` | `msh_chain_tn.ipynb` (last section) | `msh_chain_compare.ipynb` |
| `msh_compare_python.npz` | `msh_chain_python_ed.ipynb` | `msh_chain_compare.ipynb` |
| `tjunctions_python_ed_data/*.npz` | `tjunctions_python_ed.ipynb` | `tjunctions_compare.ipynb` |

Other caches and large outputs (`*.jls`, other `*.npz`, `*.npy`, `*.h5`, the figures in `*_data/`, `tb_reference/`) are not committed
(see `.gitignore`). The notebooks write them next to themselves on the first run and load them afterwards:

- `msh_chain_tn.ipynb`: `msh_chain_ldos_kpm_ramp.jls` (KPM LDOS, ~30 min), `msh_chain_charge.jls` (~10 min),
  `msh_chain_ramp.jls` (~4 min). Set the matching `RUN_*` flag to `true` to recompute. Its last section writes
  `msh_chain_tn_results.npz`.
- `msh_chain_python_ed.ipynb`: `msh_compare_python.npz`.
- `msh_tjunctions_tn.ipynb`: `msh_tjunctions_tdvp_{Z,X,F}.jls` (TDVP results, ~2.5 h with 6 threads; `RUN_TDVP = true` to recompute),
  `tdvp_progress.log` and `tdvp_status.txt` (progress of a TDVP run), `tb_reference/` (dense form of the MPO at four protocol points,
  ~21 MB, read by `tjunctions_python_ed.ipynb`), figures `nb_*.png`.
- `tjunctions_python_ed.ipynb`: `tjunctions_python_ed_data/` (LDOS, δn, gate results and timings as `.npz`, and figures).

## Running

1. **Julia notebooks.** Julia 1.13. `Project.toml` and `Manifest.toml` pin the exact package versions the outputs were
   produced with (TensorBinding 0.2.0 from the General registry, ITensors 0.9, ITensorMPS 0.3, Plots, NPZ, IJulia). From this
   directory:
   ```bash
   julia --project=. -e 'using Pkg; Pkg.instantiate()'
   julia --project=. -e 'using IJulia; installkernel("Julia MSH", "--project=@.")'
   ```
   The second line installs a Jupyter kernel that uses this project when a notebook is opened here. Select it in Jupyter
   (the notebooks' saved kernel name, `julia-1.13`, is only that of the machine they were run on).
2. **Python notebooks.** `environment.yml` (conda-forge) pins the versions used: Python 3.13, NumPy 2.3.5, SciPy 1.16.3,
   Matplotlib 3.10.6, kwant 1.5.0, numba, ipywidgets, tqdm and JupyterLab.
   ```bash
   conda env create -f environment.yml
   conda activate msh-tn
   ```
3. **MSH chain comparison.** `msh_chain_compare.ipynb` runs immediately on the committed `.npz` files and prints a ✓/✗
   table. To regenerate them, run `msh_chain_python_ed.ipynb` (about 4 minutes) and `msh_chain_tn.ipynb` (a few minutes
   with its caches), in either order, in this directory.
4. **T-junction array.** Run `msh_tjunctions_tn.ipynb` first; it writes `tb_reference/`. Its first run does the TDVP of section 6, about
   2.5 hours with six threads (start Jupyter with `JULIA_NUM_THREADS=6`; follow `tdvp_status.txt`); later runs load the cached results
   and take about 10 minutes (DMRG). Then run `tjunctions_python_ed.ipynb` (about 35 minutes) in the same directory, and finally
   `tjunctions_compare.ipynb`. The compare notebook alone runs immediately on the committed CSV and `.npz` files.

## Conventions

- **Model.** MSH chain: hopping t = 1, μ = −7, Rashba α = 0.9, s-wave Δ = 2.4, exchange J = 5.8. The phase is set by the
  spin texture θ = (−1)^(x+y)(π/2)f: f = 0 is ferromagnetic along ẑ (topological), f = 1 is the in-plane antiferromagnet
  (trivial).
- **Nambu basis.** The Julia/TensorBinding notebooks use (c↑, c↓, c†↑, c†↓) with MPO sites `[nambu; spin; pos₁ … pos_L]`.
  The Python/kwant notebooks use (c↑, c↓, c†↓, −c†↑). Spectra, LDOS, ΔQ and densities don't depend on the basis.
- **TDVP.** TensorBinding's `evolve_with_tdvp_timedep` needs `reverse_step = true`; the default gives wrong dynamics.
