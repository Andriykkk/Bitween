#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>

using namespace nvcuda;

// Simple FP16 matmul using WMMA (for comparison with quantized version)
template<int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void fp16_matmul_wmma_kernel(
    const half* __restrict__ x,     // [M, K]
    const half* __restrict__ w,     // [N, K]
    half* __restrict__ out,         // [M, N]
    int M, int N, int K
) {
    const int block_m = blockIdx.y;
    const int block_n = blockIdx.x;
    const int warp_id = threadIdx.x / 32;

    // Shared memory
    __shared__ half x_smem[WMMA_M][WMMA_K];
    __shared__ half w_smem[WMMA_N][WMMA_K];

    // WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    // Process K in chunks of WMMA_K
    for (int k_start = 0; k_start < K; k_start += WMMA_K) {

        // Load X into shared memory
        for (int idx = threadIdx.x; idx < WMMA_M * WMMA_K; idx += blockDim.x) {
            const int local_m = idx / WMMA_K;
            const int local_k = idx % WMMA_K;
            const int global_m = block_m * WMMA_M + local_m;
            const int global_k = k_start + local_k;

            if (global_m < M && global_k < K) {
                x_smem[local_m][local_k] = x[global_m * K + global_k];
            } else {
                x_smem[local_m][local_k] = __float2half(0.0f);
            }
        }

        // Load W into shared memory
        for (int idx = threadIdx.x; idx < WMMA_N * WMMA_K; idx += blockDim.x) {
            const int local_n = idx / WMMA_K;
            const int local_k = idx % WMMA_K;
            const int global_n = block_n * WMMA_N + local_n;
            const int global_k = k_start + local_k;

            if (global_n < N && global_k < K) {
                w_smem[local_n][local_k] = w[global_n * K + global_k];
            } else {
                w_smem[local_n][local_k] = __float2half(0.0f);
            }
        }

        __syncthreads();

        // Tensor core matmul
        if (warp_id == 0) {
            wmma::load_matrix_sync(a_frag, &x_smem[0][0], WMMA_K);
            wmma::load_matrix_sync(b_frag, &w_smem[0][0], WMMA_K);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Store result
    if (warp_id == 0) {
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
        for (int i = 0; i < c_frag.num_elements; ++i) {
            c_frag.x[i] = __float2half(acc_frag.x[i]);
        }

        const int out_m = block_m * WMMA_M;
        const int out_n = block_n * WMMA_N;

        if (out_m < M && out_n < N) {
            wmma::store_matrix_sync(&out[out_m * N + out_n], c_frag, N, wmma::mem_row_major);
        }
    }
}


// Naive FP16 matmul (for comparison - should be slow)
__global__ void fp16_matmul_naive_kernel(
    const half* __restrict__ x,     // [M, K]
    const half* __restrict__ w,     // [N, K]
    half* __restrict__ out,         // [M, N]
    int M, int N, int K
) {
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m < M && n < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += __half2float(x[m * K + k]) * __half2float(w[n * K + k]);
        }
        out[m * N + n] = __float2half(sum);
    }
}


extern "C" void fp16_matmul_wmma_cuda(
    const half* x,
    const half* w,
    half* out,
    int M, int N, int K
) {
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int THREADS = 32;

    dim3 grid((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
    dim3 block(THREADS);

    fp16_matmul_wmma_kernel<WMMA_M, WMMA_N, WMMA_K><<<grid, block>>>(
        x, w, out, M, N, K
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void fp16_matmul_naive_cuda(
    const half* x,
    const half* w,
    half* out,
    int M, int N, int K
) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    fp16_matmul_naive_kernel<<<grid, block>>>(x, w, out, M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}
