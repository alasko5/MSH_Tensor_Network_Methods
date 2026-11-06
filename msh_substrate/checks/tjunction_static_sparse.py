"""Sparse reference for checks/tjunction_static.jl: E_f, ΔQ and gap by shift-invert.  usage: python checks/tjunction_static_sparse.py L sabs.npz
Result 2026-11-05 (single T): E_f 3.1237201e-9, ΔQ −4.6635e-8, gap 0.458729 at 64², 128², 256² (same to 10 digits)."""
import os, sys, time, numpy as np, scipy.sparse.linalg as sl
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import tjunction_substrate as tj, substrate_model as m
L = int(sys.argv[1]); d = np.load(sys.argv[2]); Sabs = {tuple(map(int, p)): tuple(s) for p, s in zip(d["ixy"], d["S"])}
t0 = time.time(); H = tj.ham_sites(L, L, Sabs); E, V = sl.eigsh(H, k=6, sigma=0, which="LM"); Ef, dQ, gap = m.pair_values(E, V)
print(f"sparse {2**L}²: E_f {Ef:.10e} | ΔQ {dQ:+.10e} | gap {gap:.6f} | {time.time()-t0:.0f} s")
