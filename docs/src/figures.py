"""Extract the figures of the three function references from the executed notebooks.

usage: python figures.py            (run from anywhere; paths are relative to this file)

Each figure is the k-th PNG output of one notebook cell, written to <doc>/figs/<name>.png.
If a notebook is restructured, update the cell numbers here.
"""
import base64
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

KITAEV = "kitaev_charge_difference.ipynb"
FIGURES = {
    # doc: {figure name: (notebook, cell index, k-th image of that cell)}
    "kitaev": {f"cell{i}_1": (KITAEV, i, 1) for i in (11, 12, 13, 16, 22, 25, 32, 34, 40)},
    "msh_chain": {
        "tn_ldos":     ("msh_chain_tn.ipynb", 21, 1),
        "tn_kpm":      ("msh_chain_tn.ipynb", 25, 1),
        "tn_majorana": ("msh_chain_tn.ipynb", 30, 1),
        "tn_dQ":       ("msh_chain_tn.ipynb", 35, 1),
        "tn_ramp":     ("msh_chain_tn.ipynb", 45, 1),
        "py_ramp":     ("msh_chain_python_ed.ipynb", 18, 1),
        "cmp_dQ":      ("msh_chain_compare.ipynb", 9, 1),
        "cmp_ramp":    ("msh_chain_compare.ipynb", 11, 1),
        "cmp_summary": ("msh_chain_compare.ipynb", 13, 1),
    },
    "tjunctions": {
        "tn_ldos":      ("msh_tjunctions_tn.ipynb", 15, 1),
        "tn_density_Z": ("msh_tjunctions_tn.ipynb", 19, 1),
        "tn_dQ_chi":    ("msh_tjunctions_tn.ipynb", 23, 1),
        "py_dQ":        ("tjunctions_python_ed.ipynb", 18, 1),
        "py_density_X": ("tjunctions_python_ed.ipynb", 20, 2),
        "cmp_ldos":     ("tjunctions_compare.ipynb", 3, 1),
        "cmp_dn":       ("tjunctions_compare.ipynb", 5, 1),
        "cmp_scaling":  ("tjunctions_compare.ipynb", 9, 1),
        "cmp_summary":  ("tjunctions_compare.ipynb", 11, 1),
    },
}


def png_output(nb, cell, k):
    """k-th PNG output (1-based) of a notebook cell, or None."""
    pngs = [o["data"]["image/png"] for o in nb["cells"][cell].get("outputs", [])
            if "image/png" in o.get("data", {})]
    return base64.b64decode(pngs[k - 1]) if len(pngs) >= k else None


def main():
    cache, missing = {}, []
    for doc, figs in FIGURES.items():
        out = os.path.join(HERE, doc, "figs")
        os.makedirs(out, exist_ok=True)
        for name, (nbfile, cell, k) in figs.items():
            if nbfile not in cache:
                cache[nbfile] = json.load(open(os.path.join(REPO, nbfile)))
            png = png_output(cache[nbfile], cell, k)
            if png is None:
                missing.append(f"{doc}/{name}: no image {k} in {nbfile} cell {cell}")
                continue
            with open(os.path.join(out, name + ".png"), "wb") as f:
                f.write(png)
        print(f"{doc}: {len(figs)} figures -> {os.path.relpath(out, REPO)}")
    if missing:
        sys.exit("missing figures (run the notebook, or update the cell numbers):\n  " + "\n  ".join(missing))


if __name__ == "__main__":
    main()
