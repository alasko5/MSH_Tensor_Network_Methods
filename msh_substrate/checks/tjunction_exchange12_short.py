"""Sparse reference for checks/tjunction_exchange12_short.jl: midpoint expm_multiply steps (δt = 0.1) from the TN ψ(t0); 1 − F
against each TN snapshot, and the Julia texture against the Python one at t = 345.  usage: python checks/tjunction_exchange12_short.py L t0 t1"""
import os, sys, time, numpy as np
from scipy.sparse.linalg import expm_multiply
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import tjunction_substrate as tj
L, t0, t1 = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]); DT = 0.1; out = f"data/tjunction/ex12_short_{L}"
p = tj.Exchange12(); R = tj.TRamp(L, lambda t: tj.f_braid_12(p, t))
d = np.load(out + "_tex345.npz"); Sj = {tuple(map(int, a)): s for a, s in zip(d["ixy"], d["S"])}; Sp = R.S(345.0)
print("texture Julia vs Python at t = 345:", max(np.abs(np.subtract(Sj[k], Sp[k])).max() for k in Sp), len(Sj) == len(Sp))
psi = np.load(out + f"_t{round(t0)}.npy"); t = t0; ns = round(10 / DT); tt = time.time()
while t < t1 - 1e-9:
    for s in range(ns):
        psi = expm_multiply(-1j * DT * R.H(t + (s + 0.5) * DT), psi)
    t += ns * DT
    tn = np.load(out + f"_t{round(t)}.npy")
    print(f"t {t:.0f}: 1 − F = {1 - abs(np.vdot(tn, psi))**2 / np.vdot(tn, tn).real:.3e}, ‖ψ_TN‖ − 1 = {np.linalg.norm(tn) - 1:+.1e}, "
          f"{(time.time() - tt) / ns:.3f} s/step", flush=True); tt = time.time()
