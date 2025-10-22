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
        BLOCK_M = config.kwargs["BLOCK_M"]
        BLOCK_N = config.kwargs["BLOCK_N"]
        BLOCK_K = config.kwargs["BLOCK_K"]

        # Skip configs where block size exceeds matrix dimensions significantly
        if BLOCK_M > m * 2 or BLOCK_N > n * 2:
            continue

        # Skip configs with overly large K blocks for small matrices
        if k < 512 and BLOCK_K > 64:
            continue

        used.add(config)

    return list(used) if used else configs[:1]  # Keep at least one config

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=2, num_warps=4),
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
    # inputs
    x_ptr,             # (M, K) input matrix (float)
    qweight_ptr,       # (N, packed_K) int32 packed quantized weights
    scale_ptr,         # (N, num_groups) per-group scales (float)
    zp_ptr,            # (N, num_groups) per-group zero points (float)
    bias_ptr,          # (N,) bias or 0 pointer
    out_ptr,           # (M, N) output matrix (float)

    # shapes / sizes
    M, N, K,
    bits: tl.constexpr,
    qmask: tl.constexpr,  # (1<<bits)-1
    # Strides
    stride_xm, stride_xk,
    stride_qn, stride_qk,
    stride_scalen, stride_scaleg,
    stride_zpn, stride_zpg,
    stride_bias,
    stride_outm, stride_outn,

    # compile-time configs
    GROUP_SIZE: tl.constexpr,      # number of input features per group
    BLOCK_M: tl.constexpr,         # block size for M
    BLOCK_N: tl.constexpr,         # block size for N
    BLOCK_K: tl.constexpr,         # block size for K (in *unpacked* elements)
    BIAS_ENABLED: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """
    Outer product style quantized matmul kernel.

    Key optimization: Load each packed int32 ONCE, then extract all values from it.
    Uses outer product accumulation instead of tensor core matmul to avoid broadcast loads.

    Trade-off: No tensor cores, but better memory access patterns.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_off = pid_m * BLOCK_M
    n_off = pid_n * BLOCK_N

    # allocate accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=DTYPE)

    # values stored per int32
    values_per_int32: tl.constexpr = 32 // bits

    # Iterate K in blocks of BLOCK_K (unpacked elements)
    k = 0
    while k < K:
        # clamp size of this chunk
        k_chunk = tl.minimum(BLOCK_K, K - k)
        # number of packed ints to load for this k_chunk
        packed_start = k // values_per_int32
        packed_count = tl.cdiv(k_chunk, values_per_int32)

        # iterate over packed ints for this k-chunk
        for p in range(packed_count):
            # index of packed int across whole K
            packed_idx = packed_start + p

            # Load packed int32 for BLOCK_N columns: shape (BLOCK_N,)
            # This is COALESCED - each thread loads different packed int
            offs_n = n_off + tl.arange(0, BLOCK_N)
            offs_q = offs_n * stride_qn + packed_idx * stride_qk
            valid_n_mask = offs_n < N
            packed_int = tl.load(qweight_ptr + offs_q, mask=valid_n_mask, other=0)

            # For each of the sub-values inside the packed_int (values_per_int32)
            for s in range(values_per_int32):
                # compute the unpacked feature index
                unpacked_feature_idx = k + p * values_per_int32 + s

                # Only process if within bounds
                if unpacked_feature_idx < K:
                    # Load the specific x column for this feature
                    offs_m = m_off + tl.arange(0, BLOCK_M)
                    x_ptrs = offs_m * stride_xm + unpacked_feature_idx * stride_xk
                    x_col = tl.load(x_ptr + x_ptrs, mask=(offs_m < M), other=0.0)

                    # shift & mask to extract q
                    q_vals = (packed_int >> (s * bits)) & qmask    # vector length BLOCK_N (int)
                    q_f = q_vals.to(DTYPE)

                    # Compute which group this feature belongs to
                    group_idx = unpacked_feature_idx // GROUP_SIZE

                    # load per-group scale and zero_point for each output column n
                    offs_scale = offs_n * stride_scalen + group_idx * stride_scaleg
                    scale_v = tl.load(scale_ptr + offs_scale, mask=valid_n_mask, other=1.0).to(DTYPE)
                    offs_zp = offs_n * stride_zpn + group_idx * stride_zpg
                    zp_v = tl.load(zp_ptr + offs_zp, mask=valid_n_mask, other=0.0).to(DTYPE)

                    # dequantize: w = scale * (q - zero_point)
                    w_vals = scale_v * (q_f - zp_v)  # (BLOCK_N,) float

                    # Outer product update
                    acc += x_col[:, None] * w_vals[None, :]

        # advance along K
        k += k_chunk

    # Convert to output dtype
    result = acc.to(DTYPE)

    # optional: add bias per column
    if BIAS_ENABLED:
        offs_n = n_off + tl.arange(0, BLOCK_N)
        bias_n = tl.load(bias_ptr + offs_n * stride_bias, mask=(offs_n < N), other=0.0)
        result += bias_n[None, :]

    # write out block
    offs_m = m_off + tl.arange(0, BLOCK_M)
    offs_n = n_off + tl.arange(0, BLOCK_N)
    offs_out = offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_out, result, mask=mask)