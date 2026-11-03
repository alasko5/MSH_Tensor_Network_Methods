"""Model of the MSH chain on a 2D substrate, shared by msh_substrate_python.ipynb and the time-dependent notebooks.

Sparse BdG Hamiltonian (`ham`) in kwant's spinor (c↑, c↓, c†↓, −c†↑) with matrix index 4n + c, n = i_x + i_y N_x; spin
textures `S_z`, `S_tilt`; the change of Nambu basis from the Julia MPO (`julia_to_python`); E_f, ΔQ and the gap from a few
eigenpairs (`pair_values`); the second parameter point `PC` (Δ = 0.6, μ = −3, J = 3; the defaults of `ham` are the benchmark
point μ = −7, Δ = 2.4, J = 5.8); `in_fresh_process` (run a calculation in a forked process and measure its peak memory); and
`ftz_daz` / `restore_csr` (flush-to-zero and denormals-are-zero, x86 MXCSR, from ftz.c). The notebook explains each part; the
code here is the code it used to contain. The Julia counterpart is substrate_model.jl.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")    # single thread, as in the notebooks (effective only if NumPy is not imported yet)
import queue, resource, ctypes, subprocess
import multiprocessing as mp
import numpy as np
import scipy.sparse as sp

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── Pauli matrices and their products (τ ⊗ σ) ──
s0 = np.eye(2); sx = np.array([[0, 1], [1, 0]]); sy = np.array([[0, -1j], [1j, 0]]); sz = np.diag([1.0, -1.0])
tzs0, txs0 = np.kron(sz, s0), np.kron(sx, s0)
t0sx, t0sy, t0sz = np.kron(s0, sx), np.kron(s0, sy), np.kron(s0, sz)
tzsx, tzsy = np.kron(sz, sx), np.kron(sz, sy)

S_z = lambda ix: (0.0, 0.0, 1.0)
S_tilt = lambda ix: (np.sin(0.3*ix)*np.cos(0.7), np.sin(0.3*ix)*np.sin(0.7), np.cos(0.3*ix))

# ── Hamiltonian (section 1 of msh_substrate_python) ──
def shift(n):
    "n × n matrix with ones at (i + 1, i): |i⟩ → |i + 1⟩, open boundary"
    return sp.diags(np.ones(n - 1), -1, shape=(n, n), format="csr")

def chain_sites(Lx, Ly, ell):
    Nx, Ny = 2**Lx, 2**Ly
    x0, iy0 = (Nx - ell) // 2, Ny // 2
    return np.arange(x0, x0 + ell), iy0

def ham(Lx, Ly, ell, S=S_z, t=1.0, mu=-7.0, alpha=0.9, Delta=2.4, J=5.8):
    "Sparse 4N × 4N BdG Hamiltonian (CSR) of the MSH chain on the 2^Lx × 2^Ly substrate."
    Nx, Ny = 2**Lx, 2**Ly; N = Nx * Ny
    Ix, Iy = sp.identity(Nx, format="csr"), sp.identity(Ny, format="csr")
    hx = -t * tzs0 + 1j * alpha * tzsy
    hy = -t * tzs0 - 1j * alpha * tzsx
    Tx = sp.kron(sp.kron(Iy, shift(Nx)), hx)
    Ty = sp.kron(sp.kron(shift(Ny), Ix), hy)
    H = sp.kron(sp.identity(N), -mu * tzs0 + Delta * txs0) + Tx + Tx.conj().T + Ty + Ty.conj().T
    xs, iy0 = chain_sites(Lx, Ly, ell)
    for k, sk in enumerate((t0sx, t0sy, t0sz)):
        v = np.zeros(N)
        v[xs + iy0 * Nx] = [J * S(ix)[k] for ix in xs]
        if np.abs(v).max() > 1e-14:
            H = H + sp.kron(sp.diags(v), sk)
    return H.tocsr()

# ── change of Nambu basis from the Julia MPO (section 2) ──
def julia_to_python(Hj, N):
    "Julia basis (index nb·2N + σ·N + n) → Python basis (index 4n + c)."
    W = sp.lil_matrix((4 * N, 4 * N))
    for n in range(N):
        W[4*n + 0, 0*N + n] = 1            # c↑   ← (nb = 0, σ = ↑)
        W[4*n + 1, 1*N + n] = 1            # c↓   ← (nb = 0, σ = ↓)
        W[4*n + 2, 2*N + 1*N + n] = 1      # c†↓  ← (nb = 1, σ = ↓)
        W[4*n + 3, 2*N + 0*N + n] = -1     # −c†↑ ← (nb = 1, σ = ↑)
    W = W.tocsr()
    return W @ Hj @ W.T

# ── zero modes from a few eigenpairs (section 5) ──
def pair_values(E, V):
    "E_f, ΔQ of the +E member of the pair, and the gap, from a few eigenpairs around zero."
    o = np.argsort(np.abs(E)); ip = o[:2][np.argmax(E[o[:2]])]
    v = V[:, ip].reshape(-1, 4)
    return E[ip], np.sum(np.abs(v[:, :2])**2) - np.sum(np.abs(v[:, 2:])**2), np.abs(E[o[2]])

# ── the second parameter point, where the zero modes leak into the substrate (section 8) ──
PC = dict(mu=-3.0, Delta=0.6, J=3.0)

# ── helpers ──
# Every calculation that builds a large matrix runs in a forked process (in_fresh_process), so that this kernel stays small:
# a forked process starts with the kernel's resident memory, which would otherwise inflate its measured peak memory.
def in_fresh_process(fn, *args):
    "Run fn(*args) in a forked process. Returns (result or None, peak resident memory in MB, status)."
    ctx = mp.get_context("fork"); q = ctx.Queue()
    def target():
        try:
            out, status = fn(*args), "ok"
        except BaseException as e:
            out, status = None, f"{type(e).__name__}: {e}".replace(",", ";").strip()
        q.put((out, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, status))
    p = ctx.Process(target=target); p.start()
    while True:                                          # wait for the result, but notice a process that dies (e.g. killed)
        try:
            res = q.get(timeout=5); break
        except queue.Empty:
            if not p.is_alive():
                res = (None, float("nan"), f"process died (exit code {p.exitcode})"); break
    p.join()
    return res

# Flush-to-zero / denormals-are-zero (FTZ, DAZ bits of the x86 MXCSR register) for the calling thread, from ftz.c, compiled with
# cc on first use into data/libftz.so next to this file. ftz_daz(on) returns the previous MXCSR; restore_csr(c) puts it back.
FTZ_LIB = os.path.join(_HERE, "data", "libftz.so")
_ftz = None

def _ftz_lib():
    global _ftz
    if _ftz is None:
        if not os.path.isfile(FTZ_LIB):
            os.makedirs(os.path.dirname(FTZ_LIB), exist_ok=True)
            subprocess.run(["cc", "-O2", "-shared", "-fPIC", "-o", FTZ_LIB, os.path.join(_HERE, "ftz.c")], check=True)
        _ftz = ctypes.CDLL(FTZ_LIB)
        _ftz.ftz_daz.restype = ctypes.c_uint; _ftz.ftz_daz.argtypes = [ctypes.c_int]
        _ftz.restore_csr.argtypes = [ctypes.c_uint]
    return _ftz

def ftz_daz(on):
    "Switch FTZ and DAZ on (on = True) or off for this thread; returns the previous MXCSR value."
    return _ftz_lib().ftz_daz(int(bool(on)))

def restore_csr(c):
    "Restore an MXCSR value returned by ftz_daz."
    _ftz_lib().restore_csr(c)

# ── spin ramp of the chain (time-dependent notebooks) ──
# Two-sided ramp: the chain spins rotate from ẑ (topological) to the in-plane antiferromagnet (trivial), from both ends inwards,
# one site after the other (site k from the nearer end starts at t = step_T·k·rot_T and takes rot_T), until only the middle
# ell_final sites are still along ẑ: the topological segment shrinks from ell to ell_final and the two Majoranas fuse.
# S = (0, sin θ, cos θ) with θ = (−1)^ix (π/2) f, f = smoothstep(t/rot_T − step_T·k); ix is the absolute column.
RAMP_ELL_FINAL, RAMP_STEP_T = 4, 0.1

def smoothstep(t):
    "sin²(πt/2) on [0, 1], 0 before, 1 after."
    return 0.0 if t <= 0 else 1.0 if t >= 1 else np.sin(0.5 * np.pi * t) ** 2

def ramp_duration(ell, rot_T, ell_final=RAMP_ELL_FINAL, step_T=RAMP_STEP_T):
    "Time at which the last rotating site has finished."
    return (((ell - ell_final) // 2 - 1) * step_T + 1) * rot_T

def ramp_f(t, n, ell, rot_T, ell_final=RAMP_ELL_FINAL, step_T=RAMP_STEP_T):
    "Rotation fraction f ∈ [0, 1] of chain site n = 0 … ell−1 at time t."
    k = min(n, ell - 1 - n)
    return 0.0 if k >= (ell - ell_final) // 2 else smoothstep(t / rot_T - step_T * k)

def ramp_texture(t, x0, ell, rot_T, **kw):
    "Spin texture S(ix) of the chain (first column x0) at time t."
    def S(ix):
        th = (-1) ** ix * np.pi / 2 * ramp_f(t, ix - x0, ell, rot_T, **kw)
        return (0.0, np.sin(th), np.cos(th))
    return S

class Ramp:
    """H(t) of the ramp on the 2^Lx × 2^Ly substrate: the part without exchange is built once, the exchange (y and z components on
    the chain sites) is added at each call. Parameters P as for ham (default: the benchmark point)."""
    def __init__(self, Lx, Ly, ell, rot_T, ell_final=RAMP_ELL_FINAL, step_T=RAMP_STEP_T, **P):
        self.Lx, self.Ly, self.ell, self.rot_T, self.kw, self.P = Lx, Ly, ell, rot_T, dict(ell_final=ell_final, step_T=step_T), P
        self.N = 2**(Lx + Ly)
        self.xs, self.iy0 = chain_sites(Lx, Ly, ell)
        self.sites = self.xs + self.iy0 * 2**Lx
        self.J = P.get("J", 5.8)
        self.H_base = ham(Lx, Ly, ell, S=lambda ix: (0.0, 0.0, 0.0), **P)
        self.T = ramp_duration(ell, rot_T, **self.kw)

    def S(self, t):
        return ramp_texture(t, self.xs[0], self.ell, self.rot_T, **self.kw)

    def H(self, t):
        S = self.S(t); out = self.H_base
        for k, sk in ((1, t0sy), (2, t0sz)):
            v = np.zeros(self.N); v[self.sites] = [self.J * S(ix)[k] for ix in self.xs]
            out = out + sp.kron(sp.diags(v), sk)
        return out.tocsr()

def snap_every(T, dt, n=130):
    "Number of steps between recorded snapshots: about n snapshots over the ramp (the same rule in substrate_model.jl)."
    return max(1, int(round(T / dt / n)))

# ── particle–hole conjugation and Majorana readout (time-dependent notebooks) ──
def phconj(psi):
    "C ψ = τ_y σ_y ψ* in kwant's spinor (c↑, c↓, c†↓, −c†↑); C² = 1, C H C = −H (as in msh_substrate_python.ipynb, section 7)."
    v = psi.reshape(-1, 4).conj()
    return (v @ np.kron(sy, sy).T).reshape(-1)

def majoranas(psi):
    "Self-conjugate pair (γ₁, γ₂) = ((ψ + Cψ)/√2, i(ψ − Cψ)/√2) of a fermion mode ψ (orthogonal to Cψ)."
    c = phconj(psi)
    return (psi + c) / np.sqrt(2), 1j * (psi - c) / np.sqrt(2)

def majorana_matrix(phi_final, psi_T):
    """M_ij = ⟨γ_i(final pair)|γ_j(T)⟩ for the Majoranas of the final +E state φ and of the evolved ψ_f(T). Real for exact
    self-conjugate pairs; tr(MᵀM)/2 = 1 − leak, det M = p₊ − p₋, and M is orthogonal without leak. Defined up to rotations
    (the phases of φ and ψ_f)."""
    gf, gt = majoranas(phi_final), majoranas(psi_T)
    return np.array([[np.vdot(a, b) for b in gt] for a in gf])

# ── LDOS (msh_substrate_ldos_python.ipynb) ──
def site_weights(psi, Nx, Ny):
    "Particle and hole weight per site (summed over spin) of a BdG vector, as two Ny × Nx arrays (site n = ix + iy·Nx)."
    v = np.abs(psi.reshape(-1, 4))**2
    return v[:, :2].sum(1).reshape(Ny, Nx), v[:, 2:].sum(1).reshape(Ny, Nx)

def lorentz_ldos(E, w, omega, eta):
    "Σ_k w_k L_η(ω − E_k): E (k,), w (k, positions) → (len(omega), positions), Lorentzian of half-width η."
    return (eta / np.pi / ((omega[:, None] - E[None, :])**2 + eta**2)) @ w

def jackson_ldos(mu, scale, omega):
    "KPM spectral function from Chebyshev moments mu (Ncheb, …) with the Jackson kernel, in physical units (energies omega)."
    N = mu.shape[0]; n = np.arange(N); q = np.pi / (N + 1)
    g = ((N - n + 1) * np.cos(q * n) + np.sin(q * n) / np.tan(q)) / (N + 1)
    x = omega / scale; T = np.cos(np.outer(np.arccos(x), n))
    return (T * (g * np.where(n == 0, 1.0, 2.0))) @ mu / (np.pi * np.sqrt(1 - x**2))[:, None] / scale

def sparse_site_moments(H, n, Ncheb, scale, comps=(0, 1, 3, 2)):
    """Chebyshev moments ⟨n,c|T_m(H/scale)|n,c⟩ of site n with sparse products and the doubling recursion; comps in the order of
    the Julia site_kpm_moments ((0,↑), (0,↓), (1,↑), (1,↓) = Python c = 0, 1, 3, 2)."""
    Ht = (H / scale).tocsr(); mu = np.zeros((Ncheb, len(comps)))
    for j, c in enumerate(comps):
        p0 = np.zeros(Ht.shape[0], complex); p0[4 * n + c] = 1
        pm, p = p0, Ht @ p0; mu[0, j] = 1; mu[1, j] = np.vdot(p0, p).real
        for m in range(1, Ncheb // 2):
            pp = 2 * (Ht @ p) - pm
            mu[2 * m, j] = 2 * np.vdot(p, p).real - mu[0, j]; mu[2 * m + 1, j] = 2 * np.vdot(pp, p).real - mu[1, j]
            pm, p = p, pp
    return mu

def mumps_eigsh(H, k):
    "The k eigenpairs closest to zero by shift-invert eigsh with a MUMPS (METIS) factorization (ncv = 2k + 20), sorted by E."
    import mumps
    from scipy.sparse.linalg import eigsh, LinearOperator
    ctx = mumps.Context(); ctx.factor(H.tocoo(), ordering="metis")
    E, V = eigsh(H, k=k, sigma=0, which="LM", ncv=2 * k + 20, OPinv=LinearOperator(H.shape, matvec=ctx.solve, dtype=complex))
    o = np.argsort(E)
    return E[o], V[:, o]
