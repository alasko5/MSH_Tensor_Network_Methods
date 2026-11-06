"""Full-energy LDOS along a cut through the chain row: sparse Chebyshev (KPM) moments at every column of the cut.

For each site (ix, iy0) the four local states |n, c⟩ are propagated with T_{m+1} = 2 H̃ T_m − T_{m−1} (H̃ = H/scale) and the moments
⟨n,c|T_m(H̃)|n,c⟩ are saved (doubling trick: N moments from N/2 products). The Hamiltonian is built once in the parent and shared by
forked workers (copy-on-write), so the memory is about one H.

usage (from msh_substrate/):
    python checks/kpm_cut_sparse.py <bench|leak> L Ncheb scale nworkers [halfwidth=64]
writes data/ldos/kpm_cut_<point>_<L>_N<Ncheb>_s<scale>.npz with mu (columns, Ncheb, 4), the columns, x0, iy0, ell, scale, seconds.
"""
import os, sys, time, numpy as np, multiprocessing as mp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import substrate_model as m

pt, L, N, scale, nw = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
hw = int(sys.argv[6]) if len(sys.argv) > 6 else 64
P = {} if pt == "bench" else m.PC; ell = 12 if pt == "bench" else 64
Nx = 2**L; xs, iy0 = m.chain_sites(L, L, ell); x0 = int(xs[0])
cols = np.arange(max(0, x0 - hw), min(Nx, x0 + ell + hw))
t0 = time.time()
HT = (m.ham(L, L, ell, **P) / scale).tocsr()                       # shared by the forked workers
print(f"{pt} {Nx}²: H built in {time.time() - t0:.0f} s, nnz {HT.nnz}, {len(cols)} columns, {N} moments, scale {scale}", flush=True)

def site(ix):
    n = ix + iy0 * Nx; mu = np.zeros((N, 4)); t = time.time()
    for j, c in enumerate((0, 1, 3, 2)):                            # order of the Julia site_kpm_moments: (0,↑), (0,↓), (1,↑), (1,↓)
        p0 = np.zeros(HT.shape[0], complex); p0[4 * n + c] = 1
        pm, p = p0, HT @ p0; mu[0, j] = 1; mu[1, j] = p[4 * n + c].real
        for k in range(1, N // 2):
            pp = 2 * (HT @ p) - pm
            mu[2 * k, j] = 2 * np.vdot(p, p).real - mu[0, j]; mu[2 * k + 1, j] = 2 * np.vdot(pp, p).real - mu[1, j]
            pm, p = p, pp
    return ix, mu, time.time() - t

if __name__ == "__main__":
    out = f"data/ldos/kpm_cut_{pt}_{L}_N{N}_s{scale:g}.npz"
    with mp.get_context("fork").Pool(nw) as pool:
        res = {}
        for k, (ix, mu, dt) in enumerate(pool.imap_unordered(site, cols), 1):
            res[ix] = mu
            if k % max(1, len(cols) // 20) == 0 or k == len(cols):
                print(f"  {k}/{len(cols)} columns, last {dt:.0f} s per site, elapsed {time.time() - t0:.0f} s", flush=True)
    mu = np.stack([res[ix] for ix in cols])
    np.savez(out, mu=mu, cols=cols, x0=x0, iy0=iy0, ell=ell, scale=scale, seconds=time.time() - t0)
    print("written", out, flush=True)
