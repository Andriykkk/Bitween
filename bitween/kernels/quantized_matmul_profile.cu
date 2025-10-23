#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>

using namespace nvcuda;

// Version 1: Just load X (no dequantization, no WMMA)
template<int WMMA_M, int WMMA_N, int WMMA_K, int BITS>
__global__ void profile_load_x_only(
    const half* __restrict__ x,
    half* __restrict__ out,
    int M, int N, int K
) {
    const int block_m = blockIdx.y;
    const int block_n = blockIdx.x;

    __shared__ half x_smem[WMMA_M][WMMA_K];

    float dummy = 0.0f;

    for (int k_start = 0; k_start < K; k_start += WMMA_K) {
        // Load X
        for (int idx = threadIdx.x; idx < WMMA_M * WMMA_K; idx += blockDim.x) {
            const int local_m = idx / WMMA_K;
            const int local_k = idx % WMMA_K;
            const int global_m = block_m * WMMA_M + local_m;
            const int global_k = k_start + local_k;

            if (global_m < M && global_k < K) {
                x_smem[local_m][local_k] = x[global_m * K + global_k];
                dummy += __half2float(x_smem[local_m][local_k]);
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        out[0] = __float2half(dummy);
    }
}

// Version 2: Load packed weights only (no bit extraction)
template<int WMMA_M, int WMMA_N, int WMMA_K, int BITS>
__global__ void profile_load_packed_only(
    const int* __restrict__ qweight,
    half* __restrict__ out,
    int M, int N, int K
) {
    constexpr int VALUES_PER_INT32 = 32 / BITS;
    const int packed_K = K / VALUES_PER_INT32;

    const int block_m = blockIdx.y;
    const int block_n = blockIdx.x;

    int dummy = 0;

    for (int k_start = 0; k_start < K; k_start += WMMA_K) {
        // Load packed weights
        for (int idx = threadIdx.x; idx < WMMA_N * WMMA_K; idx += blockDim.x) {
            const int local_n = idx / WMMA_K;
            const int local_k = idx % WMMA_K;
            const int global_n = block_n * WMMA_N + local_n;
            const int global_k = k_start + local_k;

            if (global_n < N && global_k < K) {
                const int packed_idx = global_k / VALUES_PER_INT32;
                const int packed_val = qweight[global_n * packed_K + packed_idx];
                dummy += packed_val;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        out[0] = __float2half((float)dummy);
    }
}

// Version 3: Load + extract bits (no scale/zp, no dequant)
template<int WMMA_M, int WMMA_N, int WMMA_K, int BITS>
__global__ void profile_extract_bits_only(
    const int* __restrict__ qweight,
    half* __restrict__ out,
    int M, int N, int K
) {
    constexpr int VALUES_PER_INT32 = 32 / BITS;
    constexpr int QMASK = (1 << BITS) - 1;
    const int packed_K = K / VALUES_PER_INT32;

    const int block_m = blockIdx.y;
    const int block_n = blockIdx.x;

    int dummy = 0;

    for (int k_start = 0; k_start < K; k_start += WMMA_K) {
        for (int idx = threadIdx.x; idx < WMMA_N * WMMA_K; idx += blockDim.x) {
            const int local_n = idx / WMMA_K;
            const int local_k = idx % WMMA_K;
            const int global_n = block_n * WMMA_N + local_n;
            const int global_k = k_start + local_k;

            if (global_n < N && global_k < K) {
                const int packed_idx = global_k / VALUES_PER_INT32;
                const int bit_pos = global_k % VALUES_PER_INT32;
                const int packed_val = qweight[global_n * packed_K + packed_idx];

                // EXTRACT BITS
                const int q_val = (packed_val >> (bit_pos * BITS)) & QMASK;
                dummy += q_val;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        out[0] = __float2half((float)dummy);
    }
}

// Version 4: Load scale/zp only (no weights)
template<int WMMA_M, int WMMA_N, int WMMA_K, int BITS>
__global__ void profile_load_scale_zp_only(
    const half* __restrict__ scale,
    const half* __restrict__ zero_point,
    half* __restrict__ out,
    int M, int N, int K, int group_size
) {
    const int num_groups = (K + group_size - 1) / group_size;
    const int block_m = blockIdx.y;
    const int block_n = blockIdx.x;

    float dummy = 0.0f;

    for (int k_start = 0; k_start < K; k_start += WMMA_K) {
        for (int idx = threadIdx.x; idx < WMMA_N * WMMA_K; idx += blockDim.x) {
            const int local_n = idx / WMMA_K;
            const int local_k = idx % WMMA_K;
            const int global_n = block_n * WMMA_N + local_n;
            const int global_k = k_start + local_k;

            if (global_n < N && global_k < K) {
                // LOAD SCALE/ZP
                const int group_idx = global_k / group_size;
                const half s = scale[global_n * num_groups + group_idx];
                const half z = zero_point[global_n * num_groups + group_idx];
                dummy += __half2float(s) + __half2float(z);
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        out[0] = __float2half(dummy);
    }
}

// Version 5: Full dequantization (no WMMA)
template<int WMMA_M, int WMMA_N, int WMMA_K, int BITS>
__global__ void profile_dequant_only(
    const int* __restrict__ qweight,
    const half* __restrict__ scale,
    const half* __restrict__ zero_point,
    half* __restrict__ out,
    int M, int N, int K, int group_size
) {
    constexpr int VALUES_PER_INT32 = 32 / BITS;
    constexpr int QMASK = (1 << BITS) - 1;
    const int packed_K = K / VALUES_PER_INT32;
    const int num_groups = (K + group_size - 1) / group_size;

    const int block_m = blockIdx.y;
    const int block_n = blockIdx.x;

    __shared__ half w_smem[WMMA_N][WMMA_K];

    float dummy = 0.0f;

    for (int k_start = 0; k_start < K; k_start += WMMA_K) {
        // FULL DEQUANTIZATION
        for (int idx = threadIdx.x; idx < WMMA_N * WMMA_K; idx += blockDim.x) {
            const int local_n = idx / WMMA_K;
            const int local_k = idx % WMMA_K;
            const int global_n = block_n * WMMA_N + local_n;
            const int global_k = k_start + local_k;

            if (global_n < N && global_k < K) {
                const int packed_idx = global_k / VALUES_PER_INT32;
                const int bit_pos = global_k % VALUES_PER_INT32;
                const int packed_val = qweight[global_n * packed_K + packed_idx];
                const int q_val = (packed_val >> (bit_pos * BITS)) & QMASK;

                const int group_idx = global_k / group_size;
                const half s = scale[global_n * num_groups + group_idx];
                const half z = zero_point[global_n * num_groups + group_idx];

                const half q_f = __int2half_rn(q_val);
                w_smem[local_n][local_k] = __hmul(s, __hsub(q_f, z));
                dummy += __half2float(w_smem[local_n][local_k]);
            } else {
                w_smem[local_n][local_k] = __float2half(0.0f);
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        out[0] = __float2half(dummy);
    }
}

// Version 6: Dequant + WMMA (full kernel but simplified)
template<int WMMA_M, int WMMA_N, int WMMA_K, int BITS>
__global__ void profile_dequant_plus_wmma(
    const half* __restrict__ x,
    const int* __restrict__ qweight,
    const half* __restrict__ scale,
    const half* __restrict__ zero_point,
    half* __restrict__ out,
    int M, int N, int K, int group_size
) {
    constexpr int VALUES_PER_INT32 = 32 / BITS;
    constexpr int QMASK = (1 << BITS) - 1;
    const int packed_K = K / VALUES_PER_INT32;
    const int num_groups = (K + group_size - 1) / group_size;

    const int block_m = blockIdx.y;
    const int block_n = blockIdx.x;
    const int warp_id = threadIdx.x / 32;

    __shared__ half x_smem[WMMA_M][WMMA_K];
    __shared__ half w_smem[WMMA_N][WMMA_K];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    for (int k_start = 0; k_start < K; k_start += WMMA_K) {
        // Load X
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

        // Dequantize weights
        for (int idx = threadIdx.x; idx < WMMA_N * WMMA_K; idx += blockDim.x) {
            const int local_n = idx / WMMA_K;
            const int local_k = idx % WMMA_K;
            const int global_n = block_n * WMMA_N + local_n;
            const int global_k = k_start + local_k;

            if (global_n < N && global_k < K) {
                const int packed_idx = global_k / VALUES_PER_INT32;
                const int bit_pos = global_k % VALUES_PER_INT32;
                const int packed_val = qweight[global_n * packed_K + packed_idx];
                const int q_val = (packed_val >> (bit_pos * BITS)) & QMASK;

                const int group_idx = global_k / group_size;
                const half s = scale[global_n * num_groups + group_idx];
                const half z = zero_point[global_n * num_groups + group_idx];

                const half q_f = __int2half_rn(q_val);
                w_smem[local_n][local_k] = __hmul(s, __hsub(q_f, z));
            } else {
                w_smem[local_n][local_k] = __float2half(0.0f);
            }
        }

        __syncthreads();

        // WMMA
        if (warp_id == 0) {
            wmma::load_matrix_sync(a_frag, &x_smem[0][0], WMMA_K);
            wmma::load_matrix_sync(b_frag, &w_smem[0][0], WMMA_K);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Store
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

// Launcher functions
extern "C" {
    void profile_load_x(const half* x, half* out, int M, int N, int K) {
        constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16, THREADS = 32;
        dim3 grid((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
        dim3 block(THREADS);
        profile_load_x_only<WMMA_M, WMMA_N, WMMA_K, 8><<<grid, block>>>(x, out, M, N, K);
    }

    void profile_load_packed(const int* qweight, half* out, int M, int N, int K) {
        constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16, THREADS = 32;
        dim3 grid((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
        dim3 block(THREADS);
        profile_load_packed_only<WMMA_M, WMMA_N, WMMA_K, 8><<<grid, block>>>(qweight, out, M, N, K);
    }

    void profile_extract_bits(const int* qweight, half* out, int M, int N, int K) {
        constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16, THREADS = 32;
        dim3 grid((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
        dim3 block(THREADS);
        profile_extract_bits_only<WMMA_M, WMMA_N, WMMA_K, 8><<<grid, block>>>(qweight, out, M, N, K);
    }

    void profile_load_scale_zp(const half* scale, const half* zero_point, half* out,
                                int M, int N, int K, int group_size) {
        constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16, THREADS = 32;
        dim3 grid((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
        dim3 block(THREADS);
        profile_load_scale_zp_only<WMMA_M, WMMA_N, WMMA_K, 8><<<grid, block>>>(
            scale, zero_point, out, M, N, K, group_size);
    }

    void profile_dequant(const int* qweight, const half* scale, const half* zero_point,
                         half* out, int M, int N, int K, int group_size) {
        constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16, THREADS = 32;
        dim3 grid((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
        dim3 block(THREADS);
        profile_dequant_only<WMMA_M, WMMA_N, WMMA_K, 8><<<grid, block>>>(
            qweight, scale, zero_point, out, M, N, K, group_size);
    }

    void profile_full_kernel(const half* x, const int* qweight, const half* scale,
                             const half* zero_point, half* out,
                             int M, int N, int K, int group_size) {
        constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16, THREADS = 32;
        dim3 grid((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
        dim3 block(THREADS);
        profile_dequant_plus_wmma<WMMA_M, WMMA_N, WMMA_K, 8><<<grid, block>>>(
            x, qweight, scale, zero_point, out, M, N, K, group_size);
    }
}
