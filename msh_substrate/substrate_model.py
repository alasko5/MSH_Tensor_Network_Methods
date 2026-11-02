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
