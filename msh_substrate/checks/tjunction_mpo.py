"""Compare H_MPO ψ from checks/tjunction_mpo.jl with the sparse H ψ.  usage: python checks/tjunction_mpo.py L sabs.npz"""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import tjunction_substrate as tj
L = int(sys.argv[1]); d = np.load(sys.argv[2]); Sabs = {tuple(map(int, p)): tuple(s) for p, s in zip(d["ixy"], d["S"])}
psi, hm = np.load(f"data/tjunction/mpo_psi_{L}.npy"), np.load(f"data/tjunction/mpo_Hpsi_{L}.npy")
hs = tj.ham_sites(L, L, Sabs) @ psi
print(f"T {2**L}²: ‖(H_MPO − H_sparse)ψ‖ {np.linalg.norm(hm - hs):.2e}, ‖Hψ‖ {np.linalg.norm(hs):.2f}")
