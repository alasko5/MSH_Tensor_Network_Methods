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
