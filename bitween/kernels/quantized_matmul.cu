#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>

using namespace nvcuda;

// Optimized quantized matmul kernel using tensor cores
// Strategy: Load packed weights once, unpack to registers, dequantize to shared memory, use wmma

template<int WMMA_M, int WMMA_N, int WMMA_K, int BITS>
__global__ void quantized_matmul_wmma_kernel(
    const half* __restrict__ x,           // [M, K]
    const int* __restrict__ qweight,      // [N, packed_K]
    const half* __restrict__ scale,       // [N, num_groups]
    const half* __restrict__ zero_point,  // [N, num_groups]
    const half* __restrict__ bias,        // [N] or nullptr
    half* __restrict__ out,               // [M, N]
    int M, int N, int K,
    int group_size
) {
    constexpr int VALUES_PER_INT32 = 32 / BITS;
    constexpr int QMASK = (1 << BITS) - 1;
    const int packed_K = K / VALUES_PER_INT32;
    const int num_groups = (K + group_size - 1) / group_size;

    // Each block handles one wmma tile output (16x16)
    const int block_m = blockIdx.y;
    const int block_n = blockIdx.x;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // Shared memory for dequantized weights and input
    __shared__ half x_smem[WMMA_M][WMMA_K];
    __shared__ half w_smem[WMMA_N][WMMA_K];

    // WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    // Process K in chunks of WMMA_K
    for (int k_start = 0; k_start < K; k_start += WMMA_K) {

        // === STEP 1: Load X into shared memory (coalesced) ===
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

        // === STEP 2: Load packed weights and dequantize to shared memory ===
        // Each thread handles some columns (N dimension)
        for (int idx = threadIdx.x; idx < WMMA_N * WMMA_K; idx += blockDim.x) {
            const int local_n = idx / WMMA_K;
            const int local_k = idx % WMMA_K;
            const int global_n = block_n * WMMA_N + local_n;
            const int global_k = k_start + local_k;

            if (global_n < N && global_k < K) {
                // Calculate packed index and bit position
                const int packed_idx = global_k / VALUES_PER_INT32;
                const int bit_pos = global_k % VALUES_PER_INT32;

                // Load packed weight (single int32 load)
                const int packed_val = qweight[global_n * packed_K + packed_idx];

                // Extract quantized value
                const int q_val = (packed_val >> (bit_pos * BITS)) & QMASK;

                // Load scale and zero point (cached by group)
                const int group_idx = global_k / group_size;
                const half s = scale[global_n * num_groups + group_idx];
                const half z = zero_point[global_n * num_groups + group_idx];

                // Dequantize: w = scale * (q - zero_point)
                const half q_f = __int2half_rn(q_val);
                w_smem[local_n][local_k] = __hmul(s, __hsub(q_f, z));
            } else {
                w_smem[local_n][local_k] = __float2half(0.0f);
            }
        }

        __syncthreads();

        // === STEP 3: Tensor core matmul ===
        wmma::load_matrix_sync(a_frag, &x_smem[0][0], WMMA_K);
        wmma::load_matrix_sync(b_frag, &w_smem[0][0], WMMA_K);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

        __syncthreads();
    }

    // === STEP 4: Convert to half and store ===
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
    for (int i = 0; i < c_frag.num_elements; ++i) {
        c_frag.x[i] = __float2half(acc_frag.x[i]);
    }

    const int out_m = block_m * WMMA_M;
    const int out_n = block_n * WMMA_N;

    if (out_m < M && out_n < N) {
        wmma::store_matrix_sync(&out[out_m * N + out_n], c_frag, N, wmma::mem_row_major);
    }

    // === STEP 5: Add bias if needed ===
    if (bias != nullptr && threadIdx.x < WMMA_N) {
        const int global_n = out_n + threadIdx.x;
        if (global_n < N) {
            const half bias_val = bias[global_n];
            for (int i = 0; i < WMMA_M; ++i) {
                const int global_m = out_m + i;
                if (global_m < M) {
                    out[global_m * N + global_n] = __hadd(out[global_m * N + global_n], bias_val);
                }
            }
        }
    }
}


// Register-tiled kernel with optimized memory access
template<int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N, int BITS>
__global__ void quantized_matmul_kernel_tiled(
    const half* __restrict__ x,
    const int* __restrict__ qweight,
    const half* __restrict__ scale,
    const half* __restrict__ zero_point,
    const half* __restrict__ bias,
    half* __restrict__ out,
    int M, int N, int K,
    int group_size
) {
    constexpr int VALUES_PER_INT32 = 32 / BITS;
    constexpr int QMASK = (1 << BITS) - 1;
    const int packed_K = K / VALUES_PER_INT32;
    const int num_groups = (K + group_size - 1) / group_size;

    const int block_m = blockIdx.y;
    const int block_n = blockIdx.x;
    const int tid = threadIdx.x;

    constexpr int THREADS_PER_ROW = BLOCK_N / THREAD_N;
    const int thread_row = tid / THREADS_PER_ROW;
    const int thread_col = tid % THREADS_PER_ROW;

    // Shared memory with padding
    __shared__ half x_smem[BLOCK_M][BLOCK_K + 8];
    __shared__ half w_smem[BLOCK_N][BLOCK_K + 8];

    float acc[THREAD_M][THREAD_N];
    #pragma unroll
    for (int i = 0; i < THREAD_M; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_N; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    // Process K in chunks
    for (int k_start = 0; k_start < K; k_start += BLOCK_K) {
        const int k_chunk = min(BLOCK_K, K - k_start);

        // Load X cooperatively
        for (int idx = tid; idx < BLOCK_M * BLOCK_K; idx += blockDim.x) {
            const int local_m = idx / BLOCK_K;
            const int local_k = idx % BLOCK_K;
            const int global_m = block_m * BLOCK_M + local_m;
            const int global_k = k_start + local_k;

            if (global_m < M && global_k < K && local_k < k_chunk) {
                x_smem[local_m][local_k] = x[global_m * K + global_k];
            } else {
                x_smem[local_m][local_k] = __float2half(0.0f);
            }
        }

        // Load and dequantize weights - OPTIMIZED MEMORY ACCESS
        for (int idx = tid; idx < BLOCK_N * BLOCK_K; idx += blockDim.x) {
            const int local_n = idx / BLOCK_K;
            const int local_k = idx % BLOCK_K;
            const int global_n = block_n * BLOCK_N + local_n;
            const int global_k = k_start + local_k;

            if (global_n < N && global_k < K && local_k < k_chunk) {
                // Packed loading
                const int packed_idx = global_k / VALUES_PER_INT32;
                const int bit_pos = global_k % VALUES_PER_INT32;

                const int packed_val = qweight[global_n * packed_K + packed_idx];
                const int q_val = (packed_val >> (bit_pos * BITS)) & QMASK;

                // Cache-friendly scale/zp access
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

        // Compute with register tiling
        #pragma unroll
        for (int k = 0; k < k_chunk; ++k) {
            half x_reg[THREAD_M];
            half w_reg[THREAD_N];

            #pragma unroll
            for (int i = 0; i < THREAD_M; ++i) {
                x_reg[i] = x_smem[thread_row * THREAD_M + i][k];
            }

            #pragma unroll
            for (int j = 0; j < THREAD_N; ++j) {
                w_reg[j] = w_smem[thread_col * THREAD_N + j][k];
            }

            #pragma unroll
            for (int i = 0; i < THREAD_M; ++i) {
                #pragma unroll
                for (int j = 0; j < THREAD_N; ++j) {
                    acc[i][j] = __fmaf_rn(__half2float(x_reg[i]), __half2float(w_reg[j]), acc[i][j]);
                }
            }
        }

        __syncthreads();
    }

    // Write output
    #pragma unroll
    for (int i = 0; i < THREAD_M; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_N; ++j) {
            const int global_m = block_m * BLOCK_M + thread_row * THREAD_M + i;
            const int global_n = block_n * BLOCK_N + thread_col * THREAD_N + j;

            if (global_m < M && global_n < N) {
                float result = acc[i][j];
                if (bias != nullptr) {
                    result += __half2float(bias[global_n]);
                }
                out[global_m * N + global_n] = __float2half(result);
            }
        }
    }
}


extern "C" void quantized_matmul_cuda(
    const half* x,
    const int* qweight,
    const half* scale,
    const half* zero_point,
    const half* bias,
    half* out,
    int M, int N, int K,
    int bits, int group_size
) {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    bool use_tensor_cores = (prop.major >= 7) && (M % 16 == 0) && (N % 16 == 0) && (K % 16 == 0);

    if (use_tensor_cores) {
        // Tensor core kernel - one warp per 16x16 output tile
        constexpr int WMMA_M = 16;
        constexpr int WMMA_N = 16;
        constexpr int WMMA_K = 16;
        constexpr int THREADS = 32;  // One warp

        dim3 grid((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
        dim3 block(THREADS);

        if (bits == 2) {
            quantized_matmul_wmma_kernel<WMMA_M, WMMA_N, WMMA_K, 2><<<grid, block>>>(
                x, qweight, scale, zero_point, bias, out, M, N, K, group_size
            );
        } else if (bits == 4) {
            quantized_matmul_wmma_kernel<WMMA_M, WMMA_N, WMMA_K, 4><<<grid, block>>>(
                x, qweight, scale, zero_point, bias, out, M, N, K, group_size
            );
        } else if (bits == 8) {
            quantized_matmul_wmma_kernel<WMMA_M, WMMA_N, WMMA_K, 8><<<grid, block>>>(
                x, qweight, scale, zero_point, bias, out, M, N, K, group_size
            );
        }
    } else {
        // Register-tiled fallback
        constexpr int BLOCK_M = 64;
        constexpr int BLOCK_N = 64;
        constexpr int BLOCK_K = 32;
        constexpr int THREAD_M = 8;
        constexpr int THREAD_N = 8;
        constexpr int THREADS = (BLOCK_M / THREAD_M) * (BLOCK_N / THREAD_N);

        dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
        dim3 block(THREADS);

        if (bits == 2) {
            quantized_matmul_kernel_tiled<BLOCK_M, BLOCK_N, BLOCK_K, THREAD_M, THREAD_N, 2><<<grid, block>>>(
                x, qweight, scale, zero_point, bias, out, M, N, K, group_size
            );
        } else if (bits == 4) {
            quantized_matmul_kernel_tiled<BLOCK_M, BLOCK_N, BLOCK_K, THREAD_M, THREAD_N, 4><<<grid, block>>>(
                x, qweight, scale, zero_point, bias, out, M, N, K, group_size
            );
        } else if (bits == 8) {
            quantized_matmul_kernel_tiled<BLOCK_M, BLOCK_N, BLOCK_K, THREAD_M, THREAD_N, 8><<<grid, block>>>(
                x, qweight, scale, zero_point, bias, out, M, N, K, group_size
            );
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}
