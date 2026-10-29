#!/usr/bin/env bash
# Rebuild the function-reference PDFs in docs/ from the LaTeX sources and the executed notebooks.
#
# usage: docs/src/build.sh [kitaev] [msh_chain] [tjunctions]     (default: all three)
#
# Needs python3 (standard library only) and tectonic (XeLaTeX; e.g. `conda install -c conda-forge tectonic`).
# Tectonic downloads the TeX packages and fonts it needs on the first run.
set -euo pipefail
cd "$(dirname "$0")"

declare -A PDF=(
    [kitaev]=kitaev_charge_difference_function_reference.pdf
    [msh_chain]=msh_chain_function_reference.pdf
    [tjunctions]=msh_tjunctions_function_reference.pdf
)
docs=("$@")
[ ${#docs[@]} -eq 0 ] && docs=(kitaev msh_chain tjunctions)

python3 figures.py

for d in "${docs[@]}"; do
    [ -n "${PDF[$d]:-}" ] || { echo "unknown document: $d (kitaev, msh_chain, tjunctions)"; exit 1; }
    echo "building $d ..."
    (cd "$d" && tectonic --keep-logs main.tex > build.log 2>&1) || { grep -n '^error' "$d/build.log"; exit 1; }
    grep 'Missing character' "$d/main.log" | sort -u || true
    cp "$d/main.pdf" "../${PDF[$d]}"
    echo "  -> docs/${PDF[$d]}"
done
