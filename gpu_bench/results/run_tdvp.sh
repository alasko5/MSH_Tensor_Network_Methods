#!/bin/bash
# TDVP step runs: one Julia thread, one BLAS thread, pinned to P-core CPU 4.
cd "$(dirname "$0")"; JL=~/.julia/juliaup/julia-1.13.2+0.x64.linux.gnu/bin/julia
export JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CASES="bench9 bench10 bench11 bench11s leak128 leak256"
echo "load before: $(cat /proc/loadavg)"
taskset -c 4 $JL --project=../jlenv tdvp_bench.jl cpu $CASES
echo "load: $(cat /proc/loadavg)"
taskset -c 4 $JL --project=../jlenv tdvp_bench.jl gpu $CASES
echo "load after: $(cat /proc/loadavg)"
