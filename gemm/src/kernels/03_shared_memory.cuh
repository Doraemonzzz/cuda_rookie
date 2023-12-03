#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/*

Matrix sizes:
MxK * KxN = MxN

*/

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const uint BLOCKSIZE>
__global__ void gemm_shared_memory(int M, int N, int K, const float *A, const float *B, float *C) {
    const uint bx = blockIdx.x;
    // const uint tx = threadIdx.x % BLOCKSIZE;
    const uint tx = threadIdx.x / BLOCKSIZE;
    const uint by = blockIdx.y;
    const uint ty = threadIdx.x % BLOCKSIZE;
    // const uint ty = threadIdx.x / BLOCKSIZE;

    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    int offset_A = bx * BLOCKSIZE * K;
    int offset_B = by * BLOCKSIZE;
    int offset_C = bx * BLOCKSIZE * N + by * BLOCKSIZE;

    // start location
    A += offset_A;
    B += offset_B;
    C += offset_C;

    float tmp = 0;
    for (int k = 0; k < K; k += BLOCKSIZE) {
        // load into shared memory
        As[tx * BLOCKSIZE + ty] = A[tx * K + ty];
        Bs[tx * BLOCKSIZE + ty] = B[tx * N + ty];
        __syncthreads();
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        for (int i = 0; i < BLOCKSIZE; i++) {
            tmp += As[tx * BLOCKSIZE + i] * Bs[i * BLOCKSIZE + ty];
        }
        // ensure the compute finish before load new data
        __syncthreads();
    }

    C[tx * N + ty] = tmp;
}