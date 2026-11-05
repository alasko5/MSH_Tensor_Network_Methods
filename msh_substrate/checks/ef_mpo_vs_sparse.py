"""<ψ|H|ψ> with the sparse H of substrate_model.py for the TN zero mode saved by ef_check.jl.  args: L"""
import sys, math, numpy as np
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from substrate_model import ham
L = int(sys.argv[1])
v = np.load(f"data/efcheck/psi_{L}.npy"); nv = np.vdot(v, v).real
H = ham(L, L, 12).tocsr()
Hv = H @ v
prod = np.conj(v) * Hv
E = np.sum(prod).real / nv
E_fsum = math.fsum(prod.real) / nv
res = np.linalg.norm(Hv - E * v) / math.sqrt(nv)
tz = (np.abs(v.reshape(-1, 4)[:, :2])**2).sum() - (np.abs(v.reshape(-1, 4)[:, 2:])**2).sum()
print(f"L {L}: <ψ|H_sparse|ψ> {E:.13e}  (fsum {E_fsum:.13e})  ‖Hψ − Eψ‖ {res:.2e}  ΔQ {tz/nv:+.12e}  |ψ|²−1 {nv-1:.1e}")
