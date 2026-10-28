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
| `msh_tjunctions_tn.ipynb` | Julia | Three-T-junction MSH array: MPO Hamiltonian, subgap LDOS and Majoranas from DMRG, Z and X gates with partial-fusion readout (logical gate, ΔQ, δn(x, t)), and the full MPS/TDVP evolution compared with the exact one and with kwant. |
| `tjunctions_python_ed.ipynb` | Python | kwant/exact-diagonalization reference for `msh_tjunctions_tn.ipynb`: same geometry and protocols. Checks that kwant builds the same Hamiltonian as TensorBinding (`tb_reference/`), then the subgap LDOS, the Z and X gates with partial fusion, and δn(x, t). Writes `tjunctions_python_ed_data/`. |
| `tjunctions_compare.ipynb` | Python | Number-by-number comparison of the two T-junction notebooks: LDOS, exact δn(x, t) and the MPS/TDVP δn(x, t) and ΔQ(t) against the dense Python results, one ✓/✗ row per quantity. |
| `kitaev_charge_difference.ipynb` | Julia | Kitaev chain: fermion mode from the DMRG Majorana doublet, KPM LDOS, McWeeny bulk projector with ϵF = −(E_mode + E_gap)/2, ΔQ versus topological length, and a time-dependent smoothstep ramp that fuses the Majoranas (TDVP, with TDVP and McWeeny bulk densities). |

## Data files

Committed CSVs (small results that other notebooks read):

| File | Written by | Read by |
| --- | --- | --- |
| `msh_array_ldos_tn.csv` | `msh_tjunctions_tn.ipynb` | `tjunctions_compare.ipynb` |
| `msh_gate_density_tn_{Z,X}.csv` | `msh_tjunctions_tn.ipynb` (exact evolution) | `tjunctions_compare.ipynb` |
| `msh_gate_density_tdvp_{Z,X}.csv` | offline TDVP run (`msh_gate_density_tdvp.jl`, 1–2 h per mode) | `msh_tjunctions_tn.ipynb` (section 7), `tjunctions_compare.ipynb` |
| `msh_gate_dQ_tdvp_{Z,X}.csv` | computed from the TDVP snapshots (`tdvp_snapshots_*.jls`) | `msh_tjunctions_tn.ipynb` (section 7), `tjunctions_compare.ipynb` |

Caches and large outputs (`*.jls`, `*.npz`, `*.npy`, `*.h5`, `*_data/`, `tb_reference/`) are not committed (see `.gitignore`). The
notebooks write them next to themselves on the first run and load them afterwards:

- `msh_chain_tn.ipynb`: `msh_chain_ldos_kpm_ramp.jls` (KPM LDOS, ~30 min), `msh_chain_charge.jls` (~10 min),
  `msh_chain_ramp.jls` (~4 min). Set the matching `RUN_*` flag to `true` to recompute. Its last section writes
  `msh_chain_tn_results.npz`.
- `msh_chain_python_ed.ipynb`: `msh_compare_python.npz`.
- `msh_tjunctions_tn.ipynb`: `tb_reference/` (dense BdG matrices at four protocol points, ~21 MB, read by
  `tjunctions_python_ed.ipynb`).
- `tjunctions_python_ed.ipynb`: `tjunctions_python_ed_data/` (LDOS and δn figures and `.npz`).

## Running

1. **Julia notebooks.** Julia 1.13 with an IJulia kernel and these packages in the active environment:
   `TensorBinding` (0.2.0, General registry), `ITensors`, `ITensorMPS`, `Plots`, `NPZ`, `IJulia`.
   ```julia
   using Pkg
   Pkg.add(["TensorBinding", "ITensors", "ITensorMPS", "Plots", "NPZ", "IJulia"])
   ```
2. **Python notebooks.** A conda environment with `numpy`, `scipy`, `matplotlib`, `numba`, `ipywidgets`, `tqdm`
   and `kwant` (developed with kwant 1.5, NumPy 2.3, SciPy 1.16).
3. **MSH chain comparison.** Run `msh_chain_python_ed.ipynb` (about 4 minutes) and `msh_chain_tn.ipynb` (a few minutes
   with its caches), in either order, in this directory. Then run `msh_chain_compare.ipynb`, which loads the two `.npz` files
   and prints a ✓/✗ table.
4. **T-junction array.** Run `msh_tjunctions_tn.ipynb` first (about 14 minutes single-threaded); it writes `tb_reference/`.
   Its TDVP comparison uses the committed `msh_gate_*_tdvp_*.csv` files instead of rerunning TDVP. Then run
   `tjunctions_python_ed.ipynb` (about 12 minutes) in the same directory, and finally `tjunctions_compare.ipynb`.

## Conventions

- **Model.** MSH chain: hopping t = 1, μ = −7, Rashba α = 0.9, s-wave Δ = 2.4, exchange J = 5.8. The phase is set by the
  spin texture θ = (−1)^(x+y)(π/2)f: f = 0 is ferromagnetic along ẑ (topological), f = 1 is the in-plane antiferromagnet
  (trivial).
- **Nambu basis.** The Julia/TensorBinding notebooks use (c↑, c↓, c†↑, c†↓) with MPO sites `[nambu; spin; pos₁ … pos_L]`.
  The Python/kwant notebooks use (c↑, c↓, c†↓, −c†↑). Spectra, LDOS, ΔQ and densities don't depend on the basis.
- **TDVP.** TensorBinding's `evolve_with_tdvp_timedep` needs `reverse_step = true`; the default gives wrong dynamics.
