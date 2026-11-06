"""Time and peak memory of the static zero modes of the single T-junction against substrate size, TN against the sparse solvers
(benchmark point), with the chain on the substrate (ℓ = 12, same point, from tn_zero_modes.csv / py_zero_modes.csv) in gray for reference.
The open marker at 2048² is the MUMPS (FTZ) estimate: the 1024² time scaled by the MUMPS analysis flop ratio, and the in-core memory.

usage (from msh_substrate/): python checks/tjunction_scaling_plot.py out.png
"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

rd = lambda f: np.genfromtxt(f, delimiter=",", names=True, dtype=None, encoding=None, ndmin=1)
tn, py, est = rd("tjunction_tn_scaling.csv"), rd("tjunction_py_scaling.csv"), rd("tjunction_py_mumps_estimates.csv")
ctn, cpy = rd("tn_zero_modes.csv"), rd("py_zero_modes.csv")
ctn = ctn[(ctn["Lx"] == ctn["Ly"]) & (ctn["Lx"] >= 6)]; cpy = cpy[(cpy["Lx"] == cpy["Ly"]) & (cpy["Lx"] >= 6) & (cpy["status"] == "ok")]

C = {"TN": "#2a78d6", "superlu": "#eb6834", "mumps": "#1baf7a", "mumps_ftz": "#eda100"}           # reference palette, slots 1–4
NAME = {"TN": "TN (DMRG)", "superlu": "SuperLU", "mumps": "MUMPS", "mumps_ftz": "MUMPS, FTZ/DAZ"}
MK = {"TN": "o", "superlu": "s", "mumps": "^", "mumps_ftz": "D"}
INK, MUTED, GRID, CHAIN = "#0b0b0b", "#52514e", "#e4e3df", "#9a9993"

def series(kind):
    "(side, time s, memory GB) of one method for the T-junction."
    if kind == "TN":
        r = np.sort(tn, order="L"); return 2**r["L"], r["t_total"], r["rss_MB"] / 1e3
    r = np.sort(py[py["method"] == kind], order="L"); r = r[r["status"] == "ok"]
    return 2**r["L"], r["t"], r["rss_MB"] / 1e3

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
for k, ax in enumerate(axes):
    for kind in ("TN", "superlu", "mumps", "mumps_ftz"):
        x, t, mem = series(kind)
        ax.plot(x, (t, mem)[k], color=C[kind], lw=2, marker=MK[kind], ms=8, mec="white", mew=1.5, label=NAME[kind], zorder=3)
    # 2048² MUMPS (FTZ) estimate
    x, t, mem = series("mumps_ftz"); e = {int(r["L"]): r for r in est}
    if 10 in e and 11 in e and 1024 in x:
        t11 = t[x == 1024][0] * e[11]["est_flops"] / e[10]["est_flops"]; m11 = e[11]["est_mem_incore_MB"] / 1e3
        ax.plot([1024, 2048], [(t[x == 1024][0], mem[x == 1024][0])[k], (t11, m11)[k]], color=C["mumps_ftz"], lw=2, ls=(0, (2, 2)), zorder=2)
        ax.plot(2048, (t11, m11)[k], marker="D", ms=8, mfc="white", mec=C["mumps_ftz"], mew=2, ls="none",
                label="MUMPS, FTZ/DAZ: estimate" if k == 0 else None, zorder=3)
        ax.annotate(f"~{(t11, m11)[k]:.0f} {('s', 'GB')[k]}", (2048, (t11, m11)[k]), xytext=(-8, 8), textcoords="offset points",
                    ha="right", color=MUTED, fontsize=9)
    # chain reference
    ax.plot(2**ctn["Lx"], (ctn["t_total"], ctn["rss_MB"] / 1e3)[k], color=CHAIN, lw=1.5, ls="--", label="chain: TN", zorder=1)
    cm = np.sort(cpy[cpy["method"] == "mumps_ftz"], order="Lx")
    ax.plot(2**cm["Lx"], (cm["t"], cm["rss_MB"] / 1e3)[k], color=CHAIN, lw=1.5, ls=":", label="chain: MUMPS, FTZ/DAZ", zorder=1)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks([64, 128, 256, 512, 1024, 2048]); ax.set_xticklabels([f"{s}²" for s in (64, 128, 256, 512, 1024, 2048)])
    ax.set_xlabel("substrate size", color=MUTED); ax.set_ylabel(("time to the zero modes (s)", "peak memory (GB)")[k], color=MUTED)
    ax.set_title(("Time", "Peak memory (RSS)")[k], color=INK, loc="left", fontsize=12)
    ax.grid(True, which="major", color=GRID, lw=0.8); ax.minorticks_off()
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)
axes[0].legend(frameon=False, fontsize=9, loc="lower right")
fig.suptitle("Single T-junction on the substrate, benchmark point (one thread each); chain on the substrate in gray", color=INK, fontsize=12)
fig.savefig(sys.argv[1], dpi=110); print("written", sys.argv[1])
