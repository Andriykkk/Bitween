import triton
import triton.language as tl

def bitween_kernel_config_pruner(configs, nargs, **kwargs):
    """
    The global `configs` dict config after pruner is applied.
    """
    m, n, k = nargs["M"], nargs["N"], nargs["K"]
    
    # Prune configs that don't make sense for small matrices
    used = set()
    for config in configs:
        BLOCK_SIZE_M = config.kwargs["BLOCK_SIZE_M"]
        BLOCK_SIZE_N = config.kwargs["BLOCK_SIZE_N"]
        BLOCK_SIZE_K = config.kwargs["BLOCK_SIZE_K"]
        
        # Skip configs where block size exceeds matrix dimensions significantly
        if BLOCK_SIZE_M > m * 2 or BLOCK_SIZE_N > n * 2:
            continue
            
        # Skip configs with overly large K blocks for small matrices
        if k < 512 and BLOCK_SIZE_K > 64:
            continue
            
        used.add(config)
        
    return list(used) if used else configs[:1]  # Keep at least one config

@triton.autotune(
    configs=[
        triton.Config({
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 32,
        }, num_stages=4, num_warps=4),
        triton.Config({
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
        }, num_stages=4, num_warps=4),
        triton.Config({
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
        }, num_stages=4, num_warps=4),
        triton.Config({
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32,
        }, num_stages=4, num_warps=4),
        triton.Config({
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
        }, num_stages=4, num_warps=4),
        triton.Config({
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
        }, num_stages=2, num_warps=8),
        triton.Config({
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
        }, num_stages=2, num_warps=4),
        triton.Config({
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
        }, num_stages=3, num_warps=4),
        triton.Config({
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 128,
        }, num_stages=2, num_warps=4),
        triton.Config({
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
        }, num_stages=3, num_warps=2),
    ],
    key=["M", "N", "K"],
    prune_configs_by={
        "early_config_prune": bitween_kernel_config_pruner,
        "perf_model": None,
        "top_k": None,
    },
)
@triton.jit
def quantized_linear_kernel(
    # Pointers to input/output tensors
    x_ptr, qweight_ptr, scale_ptr, zero_point_ptr, bias_ptr, c_ptr,
    # Dimensions
    M, N, K,
    # Quantization parameters
    bits: tl.constexpr,
    maxq: tl.constexpr,
    # Strides
    stride_xm, stride_xk,
    stride_qn, stride_qk,
    stride_scale_row, stride_scale_group,
    stride_zp_row, stride_zp_group,
    stride_bias,
    stride_cm, stride_cn,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    # Other configs
    BIAS_ENABLED: tl.constexpr,
    DTYPE: tl.constexpr
):
    # 1. Block (tile) IDs
    pid_m = tl.program_id(axis=0)  # output row tile
    pid_n = tl.program_id(axis=1)  # output col tile

    # 2. Offsets in matrix
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # row indices
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # col indices

    # 3. Accumulator for result
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=DTYPE)

    values_per_int32 = 32 // bits  # e.g. 8 for 4-bit

    for k_tile in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        offs_k = k_tile * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)  # input feature idx

        # 4. Load X block (input activations)
        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        x_block = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)

        # 5. Prepare unpacked weight buffer
        deq_w_block = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=DTYPE)

        # 6. Get group index per input feature
        group_idx = offs_k // GROUP_SIZE  # shape: [BLOCK_SIZE_K]

        # 7. Bit shift for unpacking
        shift = (offs_k % values_per_int32) * bits  # shape: [BLOCK_SIZE_K]

        # 8. Compute where to load packed weights
        packed_idx = offs_k // values_per_int32  # index into packed columns
        q_ptrs = qweight_ptr + offs_n[None, :] * stride_qn + packed_idx[:, None] * stride_qk
        packed_vals = tl.load(q_ptrs, mask=(offs_n[None, :] < N), other=0)

        # 9. Unpack and dequantize
        q_vals = (packed_vals >> shift[:, None]) & maxq  # extract bits
        q_vals = q_vals.to(DTYPE)

        # 10. Load scale and zp using group indices
        scale_ptrs = scale_ptr + offs_n[None, :] * stride_scale_row + group_idx[:, None] * stride_scale_group
        zp_ptrs = zero_point_ptr + offs_n[None, :] * stride_zp_row + group_idx[:, None] * stride_zp_group
        scale = tl.load(scale_ptrs)
        zp = tl.load(zp_ptrs)

        # 11. Dequantize
        deq_w_block = (q_vals - zp.to(DTYPE)) * scale
        
        # 12. Matrix multiplication (force 16-bit output)
        dot_result = tl.dot(x_block, deq_w_block, out_dtype=DTYPE)
        acc += dot_result

    # 13. Add bias if needed
    if BIAS_ENABLED:
        bias_ptrs = bias_ptr + offs_n * stride_bias
        bias = tl.load(bias_ptrs, mask=(offs_n < N), other=0.0)
        acc += bias[None, :]

    # 14. Store result
    out_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=mask)