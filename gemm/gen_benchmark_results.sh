#!/usr/bin/env bash

set -euo pipefail

# This scripts runs the ./gemm binary for all exiting kernels, and logs
# the outputs to text files in benchmark_results/. Then it calls
# the plotting script

mkdir -p benchmark_results

for kernel in {0..1}; do
    echo ""
    ./build/gemm $kernel | tee "benchmark_results/${kernel}_output.txt"
    sleep 2
done

python plot_benchmark_results.py
