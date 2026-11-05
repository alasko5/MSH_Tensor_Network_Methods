"""Sparse step benchmark (CPU scipy expm_multiply vs GPU CuPy Taylor), reproducing idle_step of msh_substrate_td_python.ipynb
(cell 17) without FTZ/DAZ. Read-only use of msh_substrate/ (no bytecode written there).

Usage: python sparse_bench.py cpu L [L ...]          single-thread CPU step (set OMP/MKL/OPENBLAS_NUM_THREADS before running)
       python sparse_bench.py gpu L [L ...]          GPU step (H built on the CPU, moved to the GPU, exp on the GPU)
       python sparse_bench.py check                  GPU vs scipy for one step at L = 9
Appends rows to timings.csv next to this file.
"""
import sys, os, time, csv
sys.dont_write_bytecode = True                       # do not write __pycache__ into msh_substrate/
sys.path.insert(0, os.path.expanduser("~/MSH_Tensor_Network_Methods/msh_substrate"))
import numpy as np
from scipy.sparse.linalg import expm_multiply
from substrate_model import Ramp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.expanduser("~/MSH_Tensor_Network_Methods/msh_substrate/data")
DT, ELL, NST = 0.1, 12, 6
ROT = {9: 10.0, 10: 10.0, 11: 100.0, 12: 10.0}       # as in the reference IDLE_CASES (L = 11 only at rot 100)


def state(R, L, rot):
    f = os.path.join(DATA, f"td_py_psi_bench_{L}_{L}_rot{rot:.0f}.npz")
    if os.path.isfile(f):
        return np.load(f)["psiT"], "psiT"
    rng = np.random.default_rng(0); psi = rng.normal(size=4 * R.N) + 1j * rng.normal(size=4 * R.N)
    return psi / np.linalg.norm(psi), "random"


def write_rows(rows):
    f = os.path.join(HERE, "timings.csv"); new = not os.path.isfile(f)
    with open(f, "a", newline="") as io:
        w = csv.writer(io)
        if new:
            w.writerow(["method", "device", "point", "L", "rot_T", "chi", "part", "seconds"])
        w.writerows(rows)


# ── GPU exp(A) b, A = -i dt H: truncated Taylor with scaling, Al-Mohy & Higham (2011) / scipy's expm_multiply algorithm ──
# (m*, s) from the exact 1-norm of A with scipy's θ_m table (tol = 2^-53), trace shift μ = tr(A)/n, early termination of each
# Taylor series when the last two terms are below tol·‖F‖∞. This is scipy's _expm_multiply_simple_core with the exact-norm branch.
def _theta():
    from scipy.sparse.linalg._expm_multiply import _theta as th
    return th

def gpu_expm_multiply(A, b, n_trace, tol=2.0**-53):
    import cupy as cp
    n = A.shape[0]
    mu = n_trace / n
    A = A - mu * _sparse_identity(n, A.dtype)
    norm1 = float(abs(A).sum(axis=0).max())
    if norm1 == 0:
        return b, 0, 0
    th = _theta(); best = None
    for m, theta in th.items():
        s = int(np.ceil(norm1 / theta))
        if best is None or m * s < best[0] * best[1]:
            best = (m, s)
    m_star, s = best
    F = b.copy(); eta = np.exp(mu / s)
    for _ in range(s):
        c1 = float(cp.abs(b).max())
        for j in range(m_star):
            b = A.dot(b) * (1.0 / (s * (j + 1)))
            c2 = float(cp.abs(b).max())
            F += b
            if c1 + c2 <= tol * float(cp.abs(F).max()):
                break
            c1 = c2
        F *= eta
        b = F
    return F, m_star, s

def _sparse_identity(n, dtype):
    import cupyx.scipy.sparse as csp
    return csp.identity(n, dtype=dtype, format="csr")


def cpu(Ls):
    for L in Ls:
        rot = ROT[L]; R = Ramp(L, L, ELL, rot); psi, st = state(R, L, rot); t0 = 0.5 * R.T; tb, te = [], []
        for s in range(NST):
            ta = time.perf_counter(); H = R.H(t0 + (s + 0.5) * DT); tbb = time.perf_counter()
            psi = expm_multiply(-1j * DT * H, psi); tc = time.perf_counter()
            tb.append(tbb - ta); te.append(tc - tbb)
        tot = np.median(np.array(tb[1:]) + np.array(te[1:]))
        dev = f"cpu_{os.environ.get('OMP_NUM_THREADS', 'unset')}thr"
        print(f"CPU L={L} rot={rot} state={st}: H_build {np.median(tb[1:]):.3f}  expm {np.median(te[1:]):.3f}  total {tot:.3f} s", flush=True)
        write_rows([["sparse", dev, "bench", L, rot, "", "H_build", np.median(tb[1:])],
                    ["sparse", dev, "bench", L, rot, "", "step", np.median(te[1:])],
                    ["sparse", dev, "bench", L, rot, "", "total", tot]])


def gpu_step(R, H_t, psi_d):
    "build on CPU, transfer, exp on GPU; returns new psi and the three times"
    import cupy as cp, cupyx.scipy.sparse as csp
    ta = time.perf_counter(); H = R.H(H_t); A = (-1j * DT) * H; tr = complex(A.diagonal().sum()); tb = time.perf_counter()
    A_d = csp.csr_matrix(A); cp.cuda.Device().synchronize(); tc = time.perf_counter()
    psi_d, m, s = gpu_expm_multiply(A_d, psi_d, tr); cp.cuda.Device().synchronize(); td = time.perf_counter()
    return psi_d, (tb - ta, tc - tb, td - tc), (m, s)


def _dev_used():
    import cupy as cp
    free, total = cp.cuda.runtime.memGetInfo(); return (total - free) / 2**20


def gpu(Ls):
    import cupy as cp
    for L in Ls:
        rot = ROT[L]; R = Ramp(L, L, ELL, rot); psi, st = state(R, L, rot); t0 = 0.5 * R.T
        cp.get_default_memory_pool().free_all_blocks(); psi_d = cp.asarray(psi); del psi
        ts = []
        for s in range(NST):
            psi_d, t, ms = gpu_step(R, t0 + (s + 0.5) * DT, psi_d); ts.append(t)
        ts = np.array(ts[1:]); med = np.median(ts, axis=0); tot = np.median(ts.sum(1))
        mem = cp.get_default_memory_pool().total_bytes() / 2**20
        print(f"GPU L={L} rot={rot} state={st}: H_build {med[0]:.3f}  transfer {med[1]:.3f}  expm {med[2]:.3f}  total {tot:.3f} s"
              f"  (m*={ms[0]}, s={ms[1]}, nnz(H)={R.H_base.nnz}, cupy pool {mem:.0f} MiB, device used {_dev_used():.0f} MiB)", flush=True)
        write_rows([["sparse", "gpu", "bench", L, rot, "", "H_build", med[0]], ["sparse", "gpu", "bench", L, rot, "", "transfer", med[1]],
                    ["sparse", "gpu", "bench", L, rot, "", "step", med[2]], ["sparse", "gpu", "bench", L, rot, "", "total", tot]])
        del psi_d; cp.get_default_memory_pool().free_all_blocks()


def check():
    import cupy as cp
    L = 9; rot = ROT[L]; R = Ramp(L, L, ELL, rot); psi, st = state(R, L, rot); t = 0.5 * R.T + 0.5 * DT
    ref = expm_multiply(-1j * DT * R.H(t), psi)
    out, _, ms = gpu_step(R, t, cp.asarray(psi)); out = cp.asnumpy(out)
    print(f"CHECK L=9 state={st}: |psi_gpu - psi_cpu| = {np.linalg.norm(out - ref):.3e}  |psi_cpu - psi0| = {np.linalg.norm(ref - psi):.3e}"
          f"  norm(psi_gpu) - 1 = {np.linalg.norm(out) - 1:.2e}  (m*={ms[0]}, s={ms[1]})")


if __name__ == "__main__":
    {"cpu": cpu, "gpu": gpu}.get(sys.argv[1], lambda _: check())([int(x) for x in sys.argv[2:]])
