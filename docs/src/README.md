# Sources of the function references

LaTeX sources of the three PDFs in `docs/`:

| Directory | PDF |
| --- | --- |
| `kitaev/` | `kitaev_charge_difference_function_reference.pdf` |
| `msh_chain/` | `msh_chain_function_reference.pdf` |
| `tjunctions/` (`main.tex` with `partI`–`partIII.tex`) | `msh_tjunctions_function_reference.pdf` |

`preamble.tex` is shared by all three. The figures are not stored here: `figures.py` extracts them from the
outputs of the executed notebooks in the repository root (the cell numbers are listed in that script).

**Build:**

```bash
docs/src/build.sh              # all three
docs/src/build.sh msh_chain    # one of kitaev, msh_chain, tjunctions
```

This needs `python3` and [tectonic](https://tectonic-typesetting.github.io) (`conda install -c conda-forge tectonic`).
The PDFs are written to `docs/`. The numbers quoted in the text come from the notebook outputs; if you rerun a
notebook and its results change, update the corresponding `.tex` file as well.
