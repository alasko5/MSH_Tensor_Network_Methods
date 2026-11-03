"""Model of the MSH 3-T-junction array, shared by tjunctions_python_ed.ipynb and logical_gates_python_ed.ipynb.

kwant system (`build_system_3T`), spin texture and protocols (`Exchange12`, `Exchange23`, `Fusion`, ported from the Julia
code of msh_tjunctions_tn.ipynb), the BdG Hamiltonian (`ham`, `ham_sp`), the zero-mode construction of the CZ, QFT and
period-finding notebooks (`individual_MZMs`, `fix_unitary`, `zero_mode_frame`, `logical_masks`) and the sparse 2-stage
Gauss–Legendre step. The notebooks explain each part; the code here is the code they used to contain.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")          # tiny matrices: threaded BLAS only adds contention
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import warnings
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import kwant
import scipy.sparse as sp
from scipy import linalg as la
from scipy.sparse.linalg import splu

# ── Pauli matrices ──
#pauli matrices
tau_0 = np.array([[1, 0], [0, 1]])
tau_x = np.array([[0, 1], [1, 0]])
tau_y = np.array([[0, -1j], [1j, 0]])
tau_z = np.array([[1, 0], [0, -1]])

#tensor products; used in Hamiltonian function
txs0 = np.kron(tau_x, tau_0)
t0sx = np.kron(tau_0, tau_x)
t0sy = np.kron(tau_0, tau_y)
t0sz = np.kron(tau_0, tau_z)
tzsx = np.kron(tau_z, tau_x)
tzsy = np.kron(tau_z, tau_y)
tzs0 = np.kron(tau_z, tau_0)
tysy = np.kron(tau_y, tau_y)
tysx = np.kron(tau_y, tau_x)   # PHconj, spinor "uddu" (not used here: kwant's spinor is "udd-u")
tys0 = np.kron(tau_y, tau_0)   # PHconj, spinor "udud" (not used here)

# ── geometry (section 1 of tjunctions_python_ed) ──
NLEG = 16        # sites per arm (Julia TArray L = 4)
NCELL = 3
SPACING = 2 * NLEG
ARM_ANG = (0.0, np.pi, np.pi / 2)              # right, left, up
ARM_ORIG = ((0, 0), (-1, 0), (0, 1))
ARM_DIR = ((1, 0), (-1, 0), (0, 1))

def center_x(c):
    return int((c - (NCELL + 1) / 2) * SPACING)

def array_pos(c, s, n):
    return (center_x(c) + ARM_ORIG[s-1][0] + n * ARM_DIR[s-1][0],
            ARM_ORIG[s-1][1] + n * ARM_DIR[s-1][1])

label = {array_pos(c, s, n): (c, s, n) for c in range(1, NCELL + 1) for s in (1, 2, 3) for n in range(NLEG)}

def build_system_3T():
    # all sites are chain sites; kwant adds hoppings only between sites that exist
    def chain(site, mu, Delta, J, S):
        Sx, Sy, Sz = S[site.tag]
        JS = J * (Sx * t0sx + Sy * t0sy + Sz * t0sz)
        return -mu * tzs0 + Delta * txs0 + JS

    def hop_x(sitei, sitej, t, alpha):
        return -t * tzs0 + 1j * alpha * tzsy

    def hop_y(sitei, sitej, t, alpha):
        return -t * tzs0 - 1j * alpha * tzsx

    sys = kwant.Builder()
    lat = kwant.lattice.square(norbs=4)
    for (x, y) in label:
        sys[lat(x, y)] = chain
    for delta, hop in zip([(1, 0), (0, 1)], [hop_x, hop_y]):
        sys[kwant.HoppingKind(delta, lat)] = hop
    return sys.finalized()

sys = build_system_3T()
N = len(sys.sites)
tags = [tuple(int(v) for v in site.tag) for site in sys.sites]
xs = np.array([p[0] for p in tags]); ys = np.array([p[1] for p in tags])

# ── spin texture and protocols (section 2) ──
def smoothstep(t):
    return 0.0 if t <= 0 else 1.0 if t >= 1 else np.sin(0.5 * np.pi * t) ** 2

f0 = lambda c, s, n: 1.0 if (c == 2 or s == 3) else 0.0

def spins(f):
    # S[(x, y)] for the texture f(c, s, n);  θ = (−1)^(x+y) (π/2) f,  φ = θ_arm + π/2
    S = {}
    for (x, y), (c, s, n) in label.items():
        th = (-1) ** (x + y) * (np.pi / 2) * f(c, s, n)
        ph = ARM_ANG[s-1] + np.pi / 2
        S[(x, y)] = (np.cos(ph) * np.sin(th), np.sin(ph) * np.sin(th), np.cos(th))
    return S

# ── exchange_12 (L1 = L2 = 14, rot_Tz = 200, step_Tz = 0.1) on cell 1 ──
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
    # arm 1: i = L1 + 1 + n;  arm 2: i = max(L1 − n, 0);  arm 3: k = min(n, L2)
    return lambda s, n: (f_horizontal_12(p, p.L1 + 1 + n, t) if s == 1 else
                         f_horizontal_12(p, max(p.L1 - n, 0), t) if s == 2 else
                         f_vertical_12(p, min(n, p.L2), t))

def f_array_12(p, t):
    fb = f_braid_12(p, t)
    return lambda c, s, n: fb(s, n) if c == 1 else f0(c, s, n)

# ── exchange_23 (rot_Tx = 100, step_Tx = 0.1) through cell 2 ──
class Exchange23:
    def __init__(self, L1=14, L2=14, rot_T=100.0, step_T=0.1):
        self.L1, self.L2, self.rot_T, self.step_T = L1, L2, rot_T, step_T
    def duration(self):
        return (self.step_T * (self.L1 + 3 * self.L2 + 2 * self.L1 + 3) + 1) * self.rot_T

def f_h23(p, i, t):
    tau, s, L1, L2 = t / p.rot_T, p.step_T, p.L1, p.L2
    ss = smoothstep
    if 2 * L1 + 3 <= i <= 3 * L1 + 3:
        return 1 - ss(tau - s * (i - 2 * L1 - 3)) + ss(tau - s * (4 * L1 + 4 - i + L2 + L1 + 2))
    if 3 * L1 + 4 <= i <= 4 * L1 + 4:
        return 1 - ss(tau - s * (4 * L1 + 4 - i + L2 + L1 + 1)) + ss(tau - s * (i - 3 * L1 - 4 + 3 * L2 + 2 * L1 + 3))
    raise ValueError(f"i = {i} outside junction 2's segment")

def f_v23(p, k, t):
    tau, s = t / p.rot_T, p.step_T
    ss = smoothstep
    return 1 - ss(tau - s * (k + p.L1 + 2)) + ss(tau - s * (p.L2 - k + 3 * p.L1 + p.L2 + 5))

def f_array_23(p, t):
    # cell 2: right arm i = 3L1 + 3 + n; left arm (16 sites) mapped linearly onto i = 3L1 + 2 … 2L1 + 3; up k = min(n, L2)
    def f(c, s, n):
        if c != 2:
            return f0(c, s, n)
        if s == 1:
            return f_h23(p, 3 * p.L1 + 3 + n, t)
        if s == 2:
            return f_h23(p, (3 * p.L1 + 2) - (p.L1 - 1) * n / (p.L1 + 1), t)
        return f_v23(p, min(n, p.L2), t)
    return f

# ── partial fusion of pair A on cell 1 (rot_T0 = 100, step_T0 = 0.2, Lp = 1) ──
class Fusion:
    def __init__(self, L1=14, Lp=1, rot_T=100.0, step_T=0.2):
        self.L1, self.Lp, self.rot_T, self.step_T = L1, Lp, rot_T, step_T
    def duration(self):
        return ((self.L1 - self.Lp + 1) * self.step_T + 1) * self.rot_T

def f_fuse(q, t):
    tau = t / q.rot_T
    def f(c, s, n):
        if c != 1:
            return f0(c, s, n)
        if s == 3:
            return 1.0
        if s == 2 and n < q.Lp:
            return 0.0
        if s == 1 and n <= q.Lp:
            return 0.0
        return smoothstep(tau - q.step_T * (NLEG - 1 - n))     # outer end first
    return f

# ── Hamiltonian (section 3) ──
params = dict(t=1.0, mu=-7.0, alpha=0.9, Delta=2.4, J=5.8)

def ham_kwant(f, **kw):
    p = {**params, **kw, "S": spins(f)}
    return sys.hamiltonian_submatrix(params=p)

H_noJ = ham_kwant(f0, J=0.0)

def exchange_matrix(f, J=params["J"]):
    S = spins(f)
    H = np.zeros((4 * N, 4 * N), complex)
    for k, p in enumerate(tags):
        Sx, Sy, Sz = S[p]
        H[4*k:4*k+4, 4*k:4*k+4] = J * (Sx * t0sx + Sy * t0sy + Sz * t0sz)
    return H

def ham(f):
    return H_noJ + exchange_matrix(f)

def Cop(v):
    # particle–hole conjugation in the (u↑, u↓, v↓, −v↑) basis:  C = τ_y σ_y K  (C H* C = −H)
    return (np.kron(np.eye(N), tysy) @ v.conj())

H0 = ham(f0)

# ── Majoranas, fermion modes and the fixed BdG unitary (section 6) ──
def dag(A, axis1=-2, axis2=-1):
    """Return Hermitian conjugate."""
    if A.ndim > 1:
        return A.swapaxes(axis1, axis2).conj()
    else:
        return A.conj()


def PHconj(psi, ordering="ltslts", spinor="udd-u"):
    """Return the particle-hole conjugate C psi = (tau_y sigma_y) psi^* for the kwant spinor "udd-u".

    The other branches cover the spinor orderings "uddu" (tau_y sigma_x) and "udud" (tau_y sigma_0), and a
    Nambu-outer ordering "tsltsl"; this notebook uses only the default."""
    N, n = psi.shape
    if ordering == "ltslts" and spinor == "udd-u":
        return (tysy @ psi.reshape(N // 4, 4, n).conj()).reshape(N, n)
    elif ordering == "ltslts" and spinor == "uddu":
        return (tysx @ psi.reshape(N // 4, 4, n).conj()).reshape(N, n)
    elif ordering == "ltslts" and spinor == "udud":
        return (tys0 @ psi.reshape(N // 4, 4, n).conj()).reshape(N, n)
    elif ordering == "tsltsl" and spinor == "udd-u":
        tysyb = np.kron(tysy, np.eye(N//4))
        return (tysyb @ psi.conj())


def takagi(A):
    """D, U = takagi(A)

    Diagonalize a complex symmetric matrix as A = U D U^T
    where U is unitary and D is diagonal."""
    # B is Hermitian positive semi-definite
    B = dag(A) @ A
    b, V = np.linalg.eigh(B)
    # C is complex symmetric with C=X+iY, [X,Y]=0
    C = V.T @ A @ V
    X = C.real
    Y = C.imag
    # C^dag C = V^dag A^dag A V = diag(b) is real, and C^dag C = X^2 + Y^2 + i[X, Y], so [X, Y] = 0: X and Y
    # are real symmetric and commute, and one real orthogonal W diagonalizes both (eigenvectors of X + Y,
    # assuming X + Y has no accidental degeneracy). Then U^T A U = W^T C W = diag(z).
    # Simultaneously diagonalize X and Y
    w, W = np.linalg.eigh(X + Y)
    U = V @ W
    # z_i = r_i e^{i\theta_i} = [U^T A U]_{ii}
    z = np.sum(U * (A @ U), axis=0)
    r = np.abs(z)
    # Multiply by e^{-i\theta_i/2}
    U *= (r / z) ** 0.5
    # Now U^T A U = diag(r), i.e. A = U^* diag(r) U^dag: the Takagi factor in A = Z D Z^T is Z = U^*
    return r, U.conj()


def individual_MZMs(E, M, tol=1e-3, ordering="ltslts", spinor="udd-u"):
    """Fix the unitary M in place.

    Fix M so that the form
    M = [[U, V^*],
         [V, U^*]]
    is unitary
    First step: separate MZMs.

    Parameters
    ----------
    E : ndarray
        Eigenvalues.
    M : ndarray
        Eigenvectors.
    tol : float
        Max energy of zero-energy subspace.

    Returns
    -------
    mbs : ndarray
        wavefunctions with individual MZMs
    mask : ndarray
        mask pointing to zero-energy subspace
    M : ndarray
        eigenvectors

    """

    # Get zero-energy subspace
    mask = np.abs(E) < tol
    # Expand range if #states is odd
    if sum(mask) % 2 == 1:
        # Index of smallest energy outside of subspace
        ind = np.ma.array(np.abs(E), mask=mask).argmin()
        # Add index to subspace
        mask[ind] = True
        warnings.warn(f"Adding E={E[ind]} to zero-energy subspace.")

    psi = M[:, mask]
    N, n = psi.shape
    # Get Majoranas (eigenstates of C)
    # C_ij = <psi_i|C psi_j> is complex symmetric. With C = Z D Z^T (Takagi) and gamma = psi Z:
    # C gamma = (C psi) Z^* = psi C Z^* = psi Z D Z^T Z^* = gamma D, since Z^T Z^* = (Z^dag Z)^* = 1.
    # D = 1 when the subspace is closed under C, so every column of gamma is a Majorana.
    C = dag(psi) @ PHconj(psi, ordering=ordering, spinor=spinor)
    c, Z = takagi(C)
    gamma = psi @ Z

    # Localize Majoranas
    x = np.repeat(xs, 4)   # x coordinate of each component (site-major ordering, see the markdown)
    R = dag(gamma) * x @ gamma
    # R is real for Majoranas (C commutes with x), so O is real orthogonal and gamma O stays self-conjugate
    r, O = np.linalg.eigh(R.real)  # Want O ortho to conserve C
    mbs = gamma @ O

    return mbs, mask, M


def fix_unitary(mbs, mask, M):
    """Fix the unitary M in place.

    Fix M so that the form
    M = [[U, V^*],
         [V, U^*]]
    is unitary.
    second step: pair up MZMs, after manual reordering so that MZMs that you want paired are 0-1, 2-3, 4-5 etc. in the second axis of mbvs

    Parameters
    ----------
    mbs : ndarray
        wavefunctions with individual MZMs
    mask : ndarray
        mask pointing to zero-energy subspace
    M : ndarray
        eigenvectors

    Returns
    -------
    M: ndarray
        fixed unitary

    """
    N, n = mbs.shape
    # Majorana to fermion: (gamma_2k, gamma_2k+1) -> phi_k = (gamma_2k + i gamma_2k+1)/sqrt2 and C phi_k
    U = np.array([[1, 1], [1j, -1j]]) / 2**0.5
    phi = mbs @ np.kron(np.eye(n // 2), U)
    # Reorder I x tau_x -> tau_x x I: [phi_0, Cphi_0, phi_1, Cphi_1, ...] -> [phi_0, phi_1, ..., Cphi_0, Cphi_1, ...]
    ind = np.arange(n).reshape(n // 2, 2).swapaxes(0, 1).ravel()
    phi = phi[:, ind]

    # Reorder phi_{-n} = C phi_n: reverse the first half, [phi_(n/2-1), ..., phi_0, Cphi_0, ..., Cphi_(n/2-1)].
    # In the eigh ordering of a BdG matrix, column 2N+k (k-th state above zero) and column 2N-1-k (k-th below)
    # are C partners. With this order, column 2N-1-k holds phi_k and column 2N+k holds C phi_k, so the fixed M
    # keeps that mirror structure about the middle. For n = 2 the reversal does nothing.
    ind = np.arange(n).reshape(2, n // 2)
    ind[0] = np.flip(ind[0])
    ind = ind.ravel()
    phi = phi[:, ind]
    # Overwrite with corrected states
    M[:, mask] = phi

    return M


def zero_mode_frame(H):
    """Full BdG unitary of H with the fermion modes ψ_A, ψ_B (columns 2N, 2N+1) and their partners (2N−1, 2N−2).

    Returns (psi0, E, mbs, mask): the fixed unitary, the eigenvalues, the localized Majoranas and the zero-energy mask."""
    E, U = np.linalg.eigh(H)
    mbs, mask, U = individual_MZMs(E, U, tol=1e-3, ordering="ltslts", spinor="udd-u")
    # pair up Majoranas: (γ1 γ2) = ψ_A at junction 1, (γ3 γ4) = ψ_B at junction 3
    psi0 = fix_unitary(mbs[:, [0, 1, 2, 3]], mask, U).copy()
    return psi0, E, mbs, mask


def logical_masks():
    """Occupied columns of |n_A n_B⟩ for (00, 01, 10, 11): bulk below zero, occupied modes, partners of the empty ones."""
    masks = np.zeros((4, 4*N), bool)          # excited states empty
    for j, bits in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        masks[j, 2*N:2*N+2] = bits
    masks[:, :2*N] = np.flip(~masks[:, 2*N:], axis=1)   # if a mode is occupied, its particle-hole partner is empty
    return masks

# ── sparse H(t) and the 2-stage Gauss–Legendre step (sparse LU, as in the period-finding notebook) ──
H_noJ_sp = sp.csc_matrix(H_noJ)
def exchange_sparse(f, J=params["J"]):
    S = spins(f)
    return sp.block_diag([J * (S[p][0] * t0sx + S[p][1] * t0sy + S[p][2] * t0sz) for p in tags], format="csc")
def ham_sp(f):
    return (H_noJ_sp + exchange_sparse(f)).tocsc()

I_sp = sp.identity(4*N, dtype=complex, format="csc")
c1, c2 = 0.5 - 3**0.5/6, 0.5 + 3**0.5/6
a12, a21 = 0.25 - 3**0.5/6, 0.25 + 3**0.5/6
n_threads = 1      # threads for the sparse solve; 1 keeps the timings of section 11 single-thread (more is faster)
_pool = ThreadPoolExecutor(n_threads)
DT = 0.25

def step(psi, t, dt, ft):
    M0 = -1j * ham_sp(ft(t + c1*dt))
    M1 = -1j * ham_sp(ft(t + c2*dt))
    A = sp.bmat([[I_sp - 0.25*dt*M0, -a12*dt*M0],
                 [-a21*dt*M1, I_sp - 0.25*dt*M1]], format="csc")
    B = np.vstack([M0 @ psi, M1 @ psi])
    lu = splu(A, permc_spec="MMD_AT_PLUS_A")
    chunks = np.array_split(np.arange(B.shape[1]), n_threads)
    K = np.hstack(list(_pool.map(lambda c: lu.solve(np.ascontiguousarray(B[:, c])), chunks)))
    K = K.reshape(2, *psi.shape)
    return psi + 0.5*dt*(K[0] + K[1])

# ── electron density of a Slater state ──
def electron_density(W):
    """Electron number per site of the state with occupied columns W: electron part of diag(G), G = W W†."""
    return (np.abs(W) ** 2).sum(axis=1).reshape(N, 4)[:, :2].sum(axis=1)
