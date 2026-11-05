"""Compare H_MPO(t)ψ (from hcheck.jl) with the sparse H(t)ψ.  args: L"""
import sys, math, numpy as np, scipy.sparse as sp
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import substrate_model as m
L = int(sys.argv[1]); R = m.Ramp(L, L, 12, 10.0)
psi = np.load(f"data/efcheck/h_psi_{L}.npy")
Hb = R.H_base @ psi
for t in (0.0, 3.0, 6.0, 9.0, 13.0):
    S = R.S(t); hs = Hb.copy()
    for k, sk in ((1, m.t0sy), (2, m.t0sz)):
        v = np.zeros(R.N); v[R.sites] = [R.J * S(ix)[k] for ix in R.xs]
        hs += sp.kron(sp.diags(v), sk, format="csr") @ psi
    hm = np.load(f"data/efcheck/h_Hpsi_{L}_t{round(t)}.npy")
    d = hm - hs
    print(f"L {L} t {t:4.1f}: ‖(H_MPO − H_sparse)ψ‖ {np.linalg.norm(d):.3e}  ‖H ψ‖ {np.linalg.norm(hs):.3f}  "
          f"<ψ|H|ψ> sparse {np.vdot(psi, hs).real:.15e}  MPO−sparse {np.vdot(psi, d).real:+.3e}  "
          f"max |d| {np.abs(d).max():.2e} at site {np.abs(d).argmax() // 4}", flush=True)
