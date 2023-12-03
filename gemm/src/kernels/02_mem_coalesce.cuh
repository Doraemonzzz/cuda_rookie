#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/*

Matrix sizes:
MxK * KxN = MxN

*/

template <const uint BLOCKSIZE>
__global__ void gemm_mem_coalesce(int M, int N, int K, const float *A, const float *B, float *C) {
    const uint x = blockIdx.x * BLOCKSIZE + threadIdx.x / BLOCKSIZE;
    const uint y = blockIdx.y * BLOCKSIZE + threadIdx.x % BLOCKSIZE;

    if (x < M && y < N) {
        float tmp = 0;
        for (int i = 0; i < K; i++) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = tmp;
    }
}