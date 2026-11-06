"""Magnetic sites of the T-junction array on the substrate, for the Julia checks: writes ixy (sites, 2) and S (sites, 3) to an npz.
usage (from msh_substrate/): python checks/tjunction_sabs.py L ncell out.npz
ncell = 1: one T-junction, horizontal arms topological, up arm trivial; ncell = 3: the static texture f0 of the 3-T array."""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import tjunction_substrate as tj
L, ncell = int(sys.argv[1]), int(sys.argv[2])
f = (lambda c, s, n: 1.0 if s == 3 else 0.0) if ncell == 1 else tj.f0
Sabs = tj.place(tj.spins(f, ncell), L, L); k = sorted(Sabs)
np.savez(sys.argv[3], ixy=np.array(k), S=np.array([Sabs[p] for p in k]))
