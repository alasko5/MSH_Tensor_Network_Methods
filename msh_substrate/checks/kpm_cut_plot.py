"""Plot of the full-energy LDOS cut A(x, ω) along the chain row from the moments of checks/kpm_cut_sparse.py, with a subgap check.

Top: the whole spectrum from the KPM moments (Jackson kernel, resolution σ ≈ π·scale/N), summed over the four local components
(particle + hole), log scale. Bottom, below the substrate gap: with a TN levels file, the KPM against the TN levels and their partners
(levels_tn_*.npz, chain-row weights) broadened with a Gaussian of the same σ, and the largest difference and the subgap weight are
printed; with "none", the KPM from all N moments against the KPM from the first N/2 (twice the width).

usage (from msh_substrate/):
    python checks/kpm_cut_plot.py data/ldos/kpm_cut_bench_10_N1024_s16.5.npz data/ldos/levels_tn_bench_10.npz out.png 3.305
    python checks/kpm_cut_plot.py data/ldos/kpm_cut_leak_10_N2048_s9.6.npz none out.png 0.60
"""
import os, sys, numpy as np, matplotlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import substrate_model as m
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
d = np.load(sys.argv[1]); gap = float(sys.argv[4]) if len(sys.argv) > 4 else 3.305
name = os.path.basename(sys.argv[1]).split("_"); pt, L = name[2], int(name[3]); ptname = "benchmark" if pt == "bench" else "leaking"
mu, cols, x0, ell, s = d["mu"].sum(2), d["cols"], int(d["x0"]), int(d["ell"]), float(d["scale"])   # particle + hole
N = mu.shape[1]; sig = np.pi * s / N                               # Jackson resolution ≈ Gaussian σ = π s / N
omF = np.linspace(-0.99 * s, 0.99 * s, 3261); AF = m.jackson_ldos(mu.T, s, omF)        # (ω, x)
omS = np.linspace(-gap, gap, 1321); AS = m.jackson_ldos(mu.T, s, omS); ins = np.abs(omS) < (3.0 / 3.305) * gap     # |ω| < 3 at the benchmark point, the same fraction of the gap elsewhere
print(f"N={N}, scale={s}, sigma={sig:.4f}; negative min (rel) {AF.min() / AF.max():.1e}")
if sys.argv[2] != "none":
    z = np.load(sys.argv[2]); assert np.array_equal(cols, z["xcols"]) and x0 == int(z["x0"][0])
    E, row = z["E"], z["row"].sum(0)                               # TN levels, chain-row weight (x, level)
    G = lambda w: np.exp(-(omS[:, None] - w[None, :])**2 / (2 * sig**2)) / np.sqrt(2 * np.pi) / sig
    AB = G(E) @ row.T + G(-E) @ row.T; tB = f"TN eigenstates, Gaussian σ = {sig:.3f}"; tA = "KPM, subgap zoom"
    print(f"subgap |KPM−TN| max / max(TN): {np.abs(AS[ins] - AB[ins]).max() / AB[ins].max():.3f}, integrated weight |ω|<{(3.0 / 3.305) * gap:.2f}: "
          f"KPM {AS[ins].sum(0).mean() * (omS[1] - omS[0]):.4f}, TN {AB[ins].sum(0).mean() * (omS[1] - omS[0]):.4f} (per column mean)")
else:
    AB = m.jackson_ldos(mu[:, :N // 2].T, s, omS); tA = f"KPM, {N} moments (σ ≈ {sig:.3f})"; tB = f"KPM, {N // 2} moments (σ ≈ {2 * sig:.3f})"
    print(f"subgap weight |ω|<{(3.0 / 3.305) * gap:.2f} per column (mean): N {AS[ins].sum(0).mean() * (omS[1] - omS[0]):.4f}, "
          f"N/2 {AB[ins].sum(0).mean() * (omS[1] - omS[0]):.4f}")
X = cols - x0
fig = plt.figure(figsize=(14, 9), constrained_layout=True); gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])
a0 = fig.add_subplot(gs[0, :])
im = a0.pcolormesh(X, omF, np.maximum(AF, 1e-5), cmap="inferno", norm=LogNorm(1e-3, AF.max()), shading="auto")
fig.colorbar(im, ax=a0, label="A(x, ω), particle + hole")
for v in (-gap, gap): a0.axhline(v, color="c", lw=0.6, ls="--")
a0.set_ylabel("energy ω")
a0.set_title(f"Full spectrum on the chain row, {2**L}², {ptname} point: sparse KPM, {N} moments, Jackson (σ ≈ {sig:.3f}); dashed: substrate gap")
vmax = max(AS.max(), AB.max())
for k, (M, t) in enumerate(((AS, tA), (AB, tB))):
    a = fig.add_subplot(gs[1, k])
    im = a.pcolormesh(X, omS, np.maximum(M, 1e-5), cmap="inferno", norm=LogNorm(1e-3, vmax), shading="auto")
    a.set_title(t); a.set_xlabel("column ix − x₀"); a.set_ylabel("energy ω") if k == 0 else None
    if k == 1: fig.colorbar(im, ax=a)
for a in fig.axes:
    if a.get_label() != "<colorbar>" and a.get_xlabel() != "" or a is a0:
        a.axvline(-0.5, color="w", lw=0.6, ls=":"); a.axvline(ell - 0.5, color="w", lw=0.6, ls=":")
a0.set_xlabel("column ix − x₀ (chain at 0 … ℓ−1, dotted)")
fig.savefig(sys.argv[3], dpi=90); print("ok")
