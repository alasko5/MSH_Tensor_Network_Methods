"""Higher-order (HODC-type) kernel against the Jackson kernel for the per-site KPM LDOS at the first chain site, benchmark point.

Kernel of order m: K(ω, E) = Σ_l w_l · (lη/π) / ((ω − E)² + (lη)²), l = 1..m, with Σ_l w_l l^k = δ_k0 (k < m): a sum of Lorentzians of
widths η, 2η, …, mη whose combined tails cancel to order η^m (m = 1 is the Lorentzian). Its Chebyshev coefficients are obtained by
Chebyshev-Gauss quadrature (a discrete cosine transform), and the LDOS from N moments is Σ_k c_k(ω) μ_k.

Part 0: the point-spread function of one level. Part 1 (exact test, 32×32): dense eigh gives the exact local spectrum {E_j, w_j}, hence exact moments for any N and the exact
kernel-broadened LDOS (N → ∞). For each kernel and N: the truncation error, the negative dips, and how many subgap levels are resolved.
Part 2 (256²): the saved sparse moments (512) at the same site, kernels with fewer moments against Jackson with 512, with the MUMPS
subgap levels marked.

usage (from msh_substrate/): python checks/kpm_kernel_test.py   → prints the tables, writes data/ldos/kpm_kernel_test.png
"""
import os, sys, numpy as np, scipy.linalg as la
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import substrate_model as m
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SCALE = 1.05 * (7 + 5.8 + 4 * np.sqrt(1 + 0.9**2) + 2.4)          # kpm_scale() of substrate_model.jl, benchmark point
NQ = 8192                                                           # quadrature nodes for the kernel coefficients
xq = np.cos((np.arange(NQ) + 0.5) * np.pi / NQ)                     # Chebyshev nodes (rescaled energy)

def hodc_weights(order):
    l = np.arange(1, order + 1); A = np.vander(l, order, increasing=True).T     # rows k = 0..m-1: Σ_l w_l l^k
    b = np.zeros(order); b[0] = 1
    return l, np.linalg.solve(A, b)

def kernel(omega, E, eta, order):
    l, w = hodc_weights(order)
    d = omega[:, None] - E[None, :]
    return sum(wl * (ll * eta / np.pi) / (d**2 + (ll * eta)**2) for ll, wl in zip(l, w))

def kernel_coeffs(omega, eta, order, N):
    "c_k(ω), k < N, with K(ω, s·x) = Σ_k c_k(ω) T_k(x)."
    K = kernel(omega, SCALE * xq, eta, order)                        # (ω, nodes)
    T = np.cos(np.outer(np.arange(N), (np.arange(NQ) + 0.5) * np.pi / NQ))     # T_k(x_j)
    c = (2.0 / NQ) * K @ T.T; c[:, 0] /= 2
    return c

def hodc_ldos(mu, omega, eta, order):
    return kernel_coeffs(omega, eta, order, len(mu)) @ mu

def jackson(mu, omega):
    return m.jackson_ldos(mu[:, None], SCALE, omega)[:, 0]

def peaks(omega, y, levels, tol):
    "Levels with a local maximum of y within ±tol."
    mx = np.where((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]))[0] + 1
    return sum(np.any(np.abs(omega[mx] - E) < tol) for E in levels)

# ── part 0: point-spread function of a single level (FWHM and largest negative lobe, relative to the peak) ──
E0 = 1.0; om0 = np.linspace(E0 - 2.5, E0 + 2.5, 20001)
def psf(y):
    y = y / y.max(); above = np.where(y >= 0.5)[0]
    return om0[above[-1]] - om0[above[0]], max(0, -y.min())
print(f"point-spread function, single level at ω = {E0}: FWHM (largest negative lobe)")
print(f"{'N':>5s} | {'Jackson':>16s} | " + " | ".join(f"m={o} η={a}s/N".rjust(18) for o in (2, 3, 4) for a in (2, 3, 4)))
for N in (64, 128, 256, 512, 1024):
    mu = np.cos(np.arange(N) * np.arccos(E0 / SCALE)); fJ, nJ = psf(jackson(mu, om0))
    cells = [f"{f:6.3f} ({ng:6.1e})".rjust(18) for o in (2, 3, 4) for a in (2, 3, 4) for f, ng in [psf(hodc_ldos(mu, om0, a * SCALE / N, o))]]
    print(f"{N:5d} | {fJ:6.3f} ({nJ:6.1e}) | " + " | ".join(cells))
print()

# ── part 1: exact 32 × 32 ──
L = 5; H = m.ham(L, L, 12).toarray(); xs, iy0 = m.chain_sites(L, L, 12); n = xs[0] + iy0 * 2**L
E, V = la.eigh(H); w = (np.abs(V[4 * n:4 * n + 4, :])**2).sum(0)
gap = 3.305; sub = E[(np.abs(E) < gap)]; subw = w[(np.abs(E) < gap)]
lev = sub[subw > 0.01 * subw.max()]                                  # subgap levels visible at this site
mu_ex = lambda N: np.cos(np.outer(np.arange(N), np.arccos(E / SCALE))) @ w
om = np.linspace(-3.4, 3.4, 3401)
print(f"32x32, first chain site: {len(sub)} subgap levels, {len(lev)} with weight > 1 % of the largest; min spacing of those "
      f"{np.min(np.diff(np.sort(lev))):.3f}; spectral half-width {np.abs(E).max():.2f}, KPM scale {SCALE:.2f}")
print(f"{'kernel':>14s} {'N':>5s} {'eta':>7s} | {'trunc. err':>10s} {'neg. dip':>9s} {'levels resolved':>16s} {'(exact kernel)':>15s}")
rows = []
for N in (64, 128, 256, 512):
    mu = mu_ex(N)
    yJ = jackson(mu, om)
    print(f"{'Jackson':>14s} {N:5d} {np.pi * SCALE / N:7.3f} | {'—':>10s} {max(0, -yJ.min()) / yJ.max():9.1e} "
          f"{peaks(om, yJ, lev, 0.03):10d}/{len(lev):<5d}")
    rows.append(("Jackson", N, None, yJ))
    for order in (2, 3, 4):
        for a in (1.0, 2.0):                                         # η = a · s / N  (comparable to the Jackson width πs/N for a ≈ 3)
            eta = a * SCALE / N
            y = hodc_ldos(mu, om, eta, order); yinf = (kernel(om, E, eta, order) @ w)
            err = np.max(np.abs(y - yinf)) / yinf.max()
            print(f"{'order ' + str(order):>14s} {N:5d} {eta:7.3f} | {err:10.1e} {max(0, -y.min()) / y.max():9.1e} "
                  f"{peaks(om, y, lev, 0.03):10d}/{len(lev):<5d} {peaks(om, yinf, lev, 0.03):10d}")
            rows.append((f"m={order}", N, eta, y))

# ── part 2: 256², saved sparse moments ──
mu512 = np.load("data/ldos/site_kpm_bench_8_122_128_n512_m512_sparse.npy").sum(1)
lev256 = np.load("data/ldos/heat_sparse_bench_8.npz")["E"]
om2 = np.linspace(-3.4, 3.4, 3401)
yJ512 = jackson(mu512, om2)
print(f"\n256x256 (sparse moments): Jackson N=512 resolves {peaks(om2, yJ512, lev256, 0.03)}/{len(lev256)} subgap levels (all 24, any weight)")
for N in (128, 256):
    for order, a in ((3, 1.0), (3, 2.0), (4, 2.0)):
        y = hodc_ldos(mu512[:N], om2, a * SCALE / N, order)
        print(f"  order {order}, N={N}, eta={a * SCALE / N:.3f}: resolves {peaks(om2, y, lev256, 0.03)}/{len(lev256)}, "
              f"neg. dip {max(0, -y.min()) / y.max():.1e}")

# ── figure ──
fig, ax = plt.subplots(2, 1, figsize=(11, 7.5), constrained_layout=True)
yex = kernel(om, E, 0.01, 1) @ w
ax[0].plot(om, yex / yex.max(), color="0.75", lw=3, label="exact, Lorentzian η = 0.01")
for lab, N, eta, y in rows:
    if (lab, N, eta) in (("Jackson", 512, None), ("Jackson", 128, None)) or (lab == "m=3" and N == 128 and abs(eta - SCALE / 128) < 1e-9):
        ax[0].plot(om, y / y.max(), lw=1.2, label=f"{lab}, N = {N}" + (f", η = {eta:.2f}" if eta else ""))
ax[0].set_xlim(-3.4, 3.4); ax[0].set_title("32×32, first chain site: subgap LDOS (normalized)"); ax[0].legend(fontsize=8)
for E0 in lev: ax[0].axvline(E0, color="k", lw=0.4, alpha=0.4)
ax[1].plot(om2, yJ512 / yJ512.max(), color="0.6", lw=2.5, label="Jackson, N = 512")
for N, order, a in ((128, 3, 1.0), (256, 3, 1.0)):
    y = hodc_ldos(mu512[:N], om2, a * SCALE / N, order); ax[1].plot(om2, y / y.max(), lw=1.2, label=f"order {order}, N = {N}, η = {a * SCALE / N:.2f}")
for E0 in lev256: ax[1].axvline(E0, color="k", lw=0.4, alpha=0.4)
ax[1].set_xlim(-3.4, 3.4); ax[1].set_xlabel("ω"); ax[1].set_title("256², sparse moments; vertical lines: MUMPS subgap levels"); ax[1].legend(fontsize=8)
fig.savefig("data/ldos/kpm_kernel_test.png", dpi=90); print("figure: data/ldos/kpm_kernel_test.png")
