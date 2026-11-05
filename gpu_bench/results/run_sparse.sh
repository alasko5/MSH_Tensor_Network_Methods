#!/bin/bash
# Sparse step runs: single thread pinned to P-core CPU 4 (i9-14900K: CPUs 0-15 are P-core threads, 16-31 E-cores).
cd "$(dirname "$0")"; PY=~/.conda/envs/gpu_bench/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "load before: $(cat /proc/loadavg)"
taskset -c 4 $PY sparse_bench.py cpu 9 10 11
echo "load: $(cat /proc/loadavg)"
taskset -c 4 $PY sparse_bench.py gpu 9 10 11
echo "load after: $(cat /proc/loadavg)"
