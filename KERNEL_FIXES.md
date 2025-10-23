# CUDA Kernel Optimization - Applied Fixes

## Problem
Original kernel was **32x slower** than PyTorch FP16 (2.1ms vs 0.066ms)

## Root Causes Found

### 1. Only 32 Threads Per Block ❌
- **Problem**: Only 1 warp (32 threads) per block
- **Impact**: Poor GPU occupancy, wasted SM cycles
- **Fix**: Increased to 256 threads (8 warps)

### 2. Tiny Tile Size (16x16) ❌
- **Problem**: Too small to amortize overhead
- **Impact**: 32 loop iterations with 64 sync points
- **Fix**: Increased to 64x64 output tiles (16 WMMA operations per block)

### 3. Excessive Synchronization ❌
- **Problem**: 2 `__syncthreads()` per K-iteration (64 total syncs)
- **Impact**: Wasted cycles waiting
- **Fix**: Removed unnecessary second sync (now 1 per iteration)

### 4. Wrong Matrix Layout ❌
- **Problem**: Weight matrix stored row-major, but tensor cores prefer column-major for matrix B
- **Impact**: Suboptimal tensor core utilization
- **Fix**: Transposed weight storage in shared memory to column-major

### 5. Poor Work Distribution ❌
- **Problem**: With only 32 threads, each thread processed <8 elements
- **Impact**: Poor instruction-level parallelism
- **Fix**: 256 threads now process 16 elements each (better ILP)

## Applied Changes

### Configuration Changes
```cuda
// BEFORE:
constexpr int THREADS = 32;      // 1 warp
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
// Each block: 16x16 output tile

// AFTER:
constexpr int THREADS = 256;     // 8 warps
constexpr int BLOCK_M = 64;      // 64x64 output tile
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 64;      // Larger K chunks
constexpr int WMMA_TILES = 16;   // 4x4 grid of WMMA ops
```

### Memory Layout Changes
```cuda
// BEFORE:
__shared__ half w_smem[WMMA_N][WMMA_K];  // Row-major

// AFTER:
__shared__ half w_smem[BLOCK_K][BLOCK_N + 8];  // Column-major + padding
```

### Synchronization Changes
```cuda
// BEFORE:
for (int k = 0; k < K; k += WMMA_K) {
    // Load data
    __syncthreads();  // Sync 1
    // Compute
    __syncthreads();  // Sync 2 - UNNECESSARY!
}

// AFTER:
for (int k = 0; k < K; k += BLOCK_K) {
    // Load data
    __syncthreads();  // Only 1 sync
    // Compute (no second sync needed)
}
```

### Work Distribution
```cuda
// BEFORE: 32 threads loading 16x16 = 256 elements
// Each thread: 256/32 = 8 elements

// AFTER: 256 threads loading 64x64 = 4096 elements
// Each thread: 4096/256 = 16 elements (better parallelism)
```

## Expected Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Threads/block | 32 | 256 | 8x occupancy |
| Output tile | 16x16 | 64x64 | 16x work/block |
| K iterations | 32 | 8 | 4x fewer loops |
| Syncs/matmul | 64 | 16 | 4x less overhead |
| Elements/thread | 8 | 16 | 2x better ILP |

**Total theoretical speedup: 15-25x**

This should bring performance from 2.1ms to ~0.1ms, matching PyTorch FP16!

## How to Test

Run the existing profiling script:
```bash
python profile_kernel.py
```

The optimized kernel is now the default in `bitween/kernels/quantized_matmul.cu`.

## Backup

Original kernel backed up to: `bitween/kernels/quantized_matmul.cu.backup`

To revert if needed:
```bash
mv bitween/kernels/quantized_matmul.cu.backup bitween/kernels/quantized_matmul.cu
```

## Implementation Details

### Warp-Level Tile Assignment
Each of the 8 warps handles 2 WMMA tiles (16x16 each):
- Warp 0: tiles [0,0] and [0,1]
- Warp 1: tiles [0,2] and [0,3]
- Warp 2: tiles [1,0] and [1,1]
- ... and so on

This distributes the 4x4 grid of WMMA operations across all warps.

### Shared Memory Padding
Added 8-element padding to avoid bank conflicts:
```cuda
__shared__ half x_smem[BLOCK_M][BLOCK_K + 8];
__shared__ half w_smem[BLOCK_K][BLOCK_N + 8];
```

### Fragment Management
Each warp maintains arrays of fragments for its assigned tiles:
```cuda
wmma::fragment<...> a_frag[tiles_per_warp];  // 2 fragments per warp
wmma::fragment<...> b_frag[tiles_per_warp];
wmma::fragment<...> acc_frag[tiles_per_warp];
```

This allows parallel WMMA operations across warps while maintaining register locality.
