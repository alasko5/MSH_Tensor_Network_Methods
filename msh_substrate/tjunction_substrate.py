"""MSH T-junction array on the 2D superconducting substrate: geometry, spin texture and sparse BdG Hamiltonian.

The array is the one of tjunctions_model.py (3 cells, arms of NLEG = 16 sites: right, left, up; texture
θ = (−1)^(x+y) (π/2) f, φ = θ_arm + π/2), but the magnetic sites now sit on the substrate of substrate_model.py: every lattice site
carries −μ τ_z + Δ τ_x and the Rashba hoppings, and only the array sites carry the exchange J S·σ. f = 0 (S along ẑ) is the
topological segment, f = 1 (in-plane, alternating) the trivial one.

The array is placed with its horizontal row at the substrate's middle row and centred in x; the up arms point to +y.
"""
import os, sys
import numpy as np, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import substrate_model as m

# ── geometry, as in tjunctions_model.py ──
NLEG, NCELL = 16, 3
SPACING = 2 * NLEG
ARM_ANG = (0.0, np.pi, np.pi / 2)              # right, left, up
ARM_ORIG = ((0, 0), (-1, 0), (0, 1))
ARM_DIR = ((1, 0), (-1, 0), (0, 1))

def center_x(c, ncell=NCELL):
    return int((c - (ncell + 1) / 2) * SPACING)

def array_pos(c, s, n, ncell=NCELL):
    return (center_x(c, ncell) + ARM_ORIG[s-1][0] + n * ARM_DIR[s-1][0], ARM_ORIG[s-1][1] + n * ARM_DIR[s-1][1])

def labels(ncell=NCELL):
    "{(x, y) relative to the array: (cell, arm, n)}"
    return {array_pos(c, s, n, ncell): (c, s, n) for c in range(1, ncell + 1) for s in (1, 2, 3) for n in range(NLEG)}

def f0(c, s, n):
    "Static texture of the 3-T array: cells 1 and 3 horizontal arms topological, cell 2 and the up arms trivial."
    return 1.0 if (c == 2 or s == 3) else 0.0

def spins(f, ncell=NCELL):
    "{(x, y) relative: (Sx, Sy, Sz)} for the texture f(c, s, n)."
    S = {}
    for (x, y), (c, s, n) in labels(ncell).items():
        th = (-1) ** (x + y) * (np.pi / 2) * f(c, s, n)
        ph = ARM_ANG[s-1] + np.pi / 2
        S[(x, y)] = (np.cos(ph) * np.sin(th), np.sin(ph) * np.sin(th), np.cos(th))
    return S

def origin(Lx, Ly):
    "Lattice position (ix, iy) of the array point (0, 0): centred in x, horizontal row at Ny/2."
    return 2**Lx // 2, 2**Ly // 2

def place(S, Lx, Ly):
    "Relative spins → {(ix, iy): S} on the 2^Lx × 2^Ly lattice (checks that the array fits)."
    ox, oy = origin(Lx, Ly); out = {}
    for (x, y), v in S.items():
        ix, iy = ox + x, oy + y
        assert 0 <= ix < 2**Lx and 0 <= iy < 2**Ly, f"array does not fit on {2**Lx}×{2**Ly}"
        out[(ix, iy)] = v
    return out

# ── sparse Hamiltonian ──
def ham_sites(Lx, Ly, Sabs, t=1.0, mu=-7.0, alpha=0.9, Delta=2.4, J=5.8):
    """Sparse 4N × 4N BdG Hamiltonian (CSR, basis 4n + c, n = ix + iy·Nx) of the substrate with exchange J S·σ on the sites of
    Sabs = {(ix, iy): (Sx, Sy, Sz)}. With Sabs = the chain row it is substrate_model.ham."""
    H = m.ham(Lx, Ly, 0, t=t, mu=mu, alpha=alpha, Delta=Delta, J=J)          # ℓ = 0: substrate only
    Nx, N = 2**Lx, 2**(Lx + Ly)
    idx = np.array([ix + iy * Nx for (ix, iy) in Sabs]); Sv = np.array(list(Sabs.values()))
    for k, sk in enumerate((m.t0sx, m.t0sy, m.t0sz)):
        v = np.zeros(N); v[idx] = J * Sv[:, k]
        if np.abs(v).max() > 1e-14:
            H = H + sp.kron(sp.diags(v), sk)
    return H.tocsr()

# ── exchange_12 on one T-junction (cell 1), as in tjunctions_model.py (L1 = L2 = 14, rot_T = 200, step_T = 0.1) ──
def smoothstep(t):
    return 0.0 if t <= 0 else 1.0 if t >= 1 else np.sin(0.5 * np.pi * t) ** 2

class Exchange12:
    def __init__(self, L1=14, L2=14, rot_T=200.0, step_T=0.1):
        self.L1, self.L2, self.rot_T, self.step_T = L1, L2, rot_T, step_T
    def duration(self):
        return (self.step_T * (4 * self.L1 + 2 * self.L2 + 8) + 1) * self.rot_T

def f_horizontal_12(p, i, t):
    tau, s, L1, L2 = t / p.rot_T, p.step_T, p.L1, p.L2
    ss = smoothstep
    if 0 <= i <= L1:
        return ss(tau - s * (i + L1 + L2 + 2)) - ss(tau - s * (L1 - i + 3 * L1 + 2 * L2 + 8))
    if i == L1 + 1:
        return 0.0
    if L1 + 2 <= i <= 2 * L1 + 2:
        return ss(tau - s * (2 * L1 + 2 - i)) - ss(tau - s * (i + L1 + L2 + 2))
    return 1.0

def f_vertical_12(p, k, t):
    tau, s, L1, L2 = t / p.rot_T, p.step_T, p.L1, p.L2
    ss = smoothstep
    return 1 - ss(tau - s * (k + L1 + 2)) + ss(tau - s * (L2 - k + 3 * L1 + L2 + 5))

def f_braid_12(p, t):
    "Texture f(c, s, n) of exchange_12 at time t (arm 1: i = L1 + 1 + n; arm 2: i = max(L1 − n, 0); arm 3: k = min(n, L2))."
    return lambda c, s, n: (f_horizontal_12(p, p.L1 + 1 + n, t) if s == 1 else
                            f_horizontal_12(p, max(p.L1 - n, 0), t) if s == 2 else
                            f_vertical_12(p, min(n, p.L2), t))

class TRamp:
    """H(t) of a T-junction protocol on the 2^L × 2^L substrate: the substrate part is built once, the exchange on the array sites is
    added at each call. ftex(t) gives the texture f(c, s, n); ncell as in labels()."""
    def __init__(self, L, ftex, ncell=1, **P):
        self.L, self.ftex, self.ncell, self.P = L, ftex, ncell, P
        self.J = P.get("J", 5.8); self.Nx = 2**L; self.N = 4**L
        self.H_base = m.ham(L, L, 0, **P)
        self.keys = sorted(place(spins(ftex(0.0), ncell), L, L))
        self.idx = np.array([ix + iy * self.Nx for (ix, iy) in self.keys])
    def S(self, t):
        return place(spins(self.ftex(t), self.ncell), self.L, self.L)
    def H(self, t):
        S = self.S(t); Sv = np.array([S[k] for k in self.keys]); out = self.H_base
        for k, sk in enumerate((m.t0sx, m.t0sy, m.t0sz)):
            v = np.zeros(self.N); v[self.idx] = self.J * Sv[:, k]
            out = out + sp.kron(sp.diags(v), sk)
        return out.tocsr()
