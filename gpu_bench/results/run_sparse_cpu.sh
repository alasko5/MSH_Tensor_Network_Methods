#!/bin/bash
# Repeat of the single-thread CPU sparse step (pinned to P-core CPU 4), then all cores (threads unrestricted, no pinning).
cd "$(dirname "$0")"; PY=~/.conda/envs/gpu_bench/bin/python
echo "load before: $(cat /proc/loadavg)"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 4 $PY sparse_bench.py cpu 9 10 11
echo "load: $(cat /proc/loadavg)"
OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 $PY sparse_bench.py cpu 9 10 11
echo "load after: $(cat /proc/loadavg)"
