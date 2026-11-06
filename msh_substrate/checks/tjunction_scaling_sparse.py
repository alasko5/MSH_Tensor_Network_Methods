"""Static scaling of the single T-junction on the substrate, sparse side: zero modes by shift-invert eigsh with SuperLU, MUMPS and MUMPS
with flush-to-zero (FTZ/DAZ), single thread, each run in a fresh forked process (peak RSS), as in section 5 of msh_substrate_python.ipynb;
and the MUMPS analysis estimates (factor flops and in-core memory) up to 2048², for the sizes that are not run.

usage (from msh_substrate/): python checks/tjunction_scaling_sparse.py <bench|leak> Lmax [Lmin=6]
appends to tjunction_py_scaling.csv and tjunction_py_mumps_estimates.csv (rows already present are skipped).
Result 2026-11-06 (benchmark point, each size alone): MUMPS (FTZ) 0.4 s at 64² … 285 s / 24.9 GB at 1024², est. ~2100 s / 114 GB at
2048²; equal to TN in time at ≈ 790², in memory at ≈ 330². The *_leak16.csv files: leaking point with 16-site arms, 64² only (on hold).
"""
import os, sys, time
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
import numpy as np
from scipy.sparse.linalg import eigsh, splu, LinearOperator
import mumps
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import tjunction_substrate as tj, substrate_model as m

pt, Lmax = sys.argv[1], int(sys.argv[2]); Lmin = int(sys.argv[3]) if len(sys.argv) > 3 else 6
P = {} if pt == "bench" else m.PC
OUT, EST = "tjunction_py_scaling.csv", "tjunction_py_mumps_estimates.csv"
COLS = ["point", "L", "N", "D", "method", "t", "rss_MB", "E_f", "dQ", "E_gap", "status"]
ECOLS = ["point", "L", "N", "D", "est_nonzeros", "est_flops", "est_mem_incore_MB", "est_mem_ooc_MB", "t_analysis"]

def ham(L):
    "Single T (horizontal arms topological, up arm trivial) at the array position of tjunction_substrate.origin."
    return tj.ham_sites(L, L, tj.place(tj.spins(lambda c, s, n: 1.0 if s == 3 else 0.0, 1), L, L), **P)

def solve(L, method):
    H = ham(L)
    if method == "superlu":
        H = H.tocsc(); t_ = time.perf_counter()
        try:
            lu = splu(H, permc_spec="MMD_AT_PLUS_A")
        except MemoryError:
            raise MemoryError("SuperLU: not enough memory to perform factorization") from None
        E, V = eigsh(H, k=4, sigma=0, which="LM", OPinv=LinearOperator(H.shape, matvec=lu.solve, dtype=complex))
    else:
        csr = m.ftz_daz(method == "mumps_ftz")
        try:
            t_ = time.perf_counter()
            ctx = mumps.Context(); ctx.factor(H.tocoo(), ordering="metis")
            E, V = eigsh(H, k=4, sigma=0, which="LM", OPinv=LinearOperator(H.shape, matvec=ctx.solve, dtype=complex))
        finally:
            m.restore_csr(csr)
    return (time.perf_counter() - t_, *m.pair_values(E, V))

def estimate(L):
    H = ham(L).tocoo(); ctx = mumps.Context(); t_ = time.perf_counter(); ctx.analyze(H, ordering="metis")
    s = ctx.analysis_stats
    return (pt, L, 4**L, 4 * 4**L, s.est_nonzeros, s.est_flops, s.est_mem_incore, s.est_mem_ooc, time.perf_counter() - t_)

def done(fn, key):
    if not os.path.isfile(fn):
        with open(fn, "w") as f:
            f.write(",".join(COLS if fn == OUT else ECOLS) + "\n")
        return set()
    r = np.genfromtxt(fn, delimiter=",", names=True, dtype=None, encoding=None, ndmin=1)
    return {key(x) for x in r}

if __name__ == "__main__":
    for meth in ("superlu", "mumps", "mumps_ftz"):                            # warm-up
        solve(6, meth)
    have = done(OUT, lambda r: (str(r["point"]), int(r["L"]), str(r["method"])))
    for L in range(Lmin, Lmax + 1):
        for meth in ("superlu", "mumps", "mumps_ftz"):
            if (pt, L, meth) in have:
                continue
            out, rss, status = m.in_fresh_process(solve, L, meth)
            t, Ef, dQ, gap = out if out is not None else (np.nan,) * 4
            with open(OUT, "a") as f:
                f.write(",".join(str(x) for x in (pt, L, 4**L, 4 * 4**L, meth, t, rss, Ef, dQ, gap, status)) + "\n")
            print(f"{pt} {2**L}² {meth:9s}: {t:8.1f} s, {rss:8.0f} MB, E_f {Ef:.6e}, ΔQ {dQ:+.4e}, gap {gap:.6f}  {status}", flush=True)
    have = done(EST, lambda r: (str(r["point"]), int(r["L"])))
    for L in range(7, 12):
        if (pt, L) in have:
            continue
        r = m.in_fresh_process(estimate, L)[0]
        with open(EST, "a") as f:
            f.write(",".join(str(x) for x in r) + "\n")
        print(f"{pt} {2**L}² MUMPS estimate: {r[5]:.2e} flops, {r[6]:.0f} MB in core", flush=True)
