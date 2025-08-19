import triton
import triton.language as tl

@triton.jit
def quantized_linear_kernel2(
    # Pointers to input/output tensors
    x_ptr, qweight_ptr, scale_ptr, zero_point_ptr, bias_ptr, c_ptr,
    # Dimensions
    M, N, K,
    # Strides
    stride_xm, stride_xk,
    stride_qn, stride_qk,
    stride_sm,
    stride_zm,
    stride_bm,
    stride_cm, stride_cn,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    # Other configs
    BIAS_ENABLED: tl.constexpr
):
    """
    Triton kernel for quantized linear forward pass.
    C = X @ dequantize(Q_W) + B
    
    This kernel dequantizes the weights on the fly during the matmul operation.
    It supports 8-bit quantization with per-row scaling.
    """
    # Get program IDs for the output block
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Offsets for the current block of C
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Accumulator for the result of the block, initialized to zero
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the inner K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        offs_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        
        # Load block of input activation X
        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        x_block = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K)).to(tl.float16)

        # Load block of quantized weight Q_W
        q_w_ptrs = qweight_ptr + (offs_n[None, :] * stride_qn + offs_k[:, None] * stride_qk)
        q_w_block = tl.load(q_w_ptrs, mask=(offs_n[None, :] < N) & (offs_k[:, None] < K))
        
        # --- FIX: Load scale and zero point as 1D tensors ---
        # The previous code was incorrectly loading them as 2D tensors.
        # This caused the dequantized_w_block to become 3D, leading to the dot product error.
        scale_ptrs = scale_ptr + offs_n * stride_sm
        scale_block = tl.load(scale_ptrs, mask=(offs_n < N))
        
        # Correctly load the zero point from its own tensor
        zero_point_ptrs = zero_point_ptr + offs_n * stride_zm
        zero_point_block = tl.load(zero_point_ptrs, mask=(offs_n < N))

        # Dequantize the weight block on the fly
        # The `[None, :]` is needed here to correctly broadcast the 1D vectors
        # (scale_block, zero_point_block) across the K dimension of the weight block.
        dequantized_w_block = (q_w_block.to(tl.float16) - zero_point_block.to(tl.float16)[None, :]) * scale_block.to(tl.float16)[None, :]
        
        # Perform the block-wise matrix multiplication
        accumulator += tl.dot(x_block, dequantized_w_block)
    
    # Load bias if enabled and add it to the accumulator
    if BIAS_ENABLED:
        bias_ptrs = bias_ptr + offs_n * stride_bm
        bias_block = tl.load(bias_ptrs, mask=(offs_n < N), other=0.0)
        accumulator += bias_block[None, :]

    # Store the final result back to global memory
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, accumulator, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


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
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

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
        q_vals = q_vals.to(tl.float32)

        # 10. Load scale and zp using group indices
        scale_ptrs = scale_ptr + offs_n[None, :] * stride_scale_row + group_idx[:, None] * stride_scale_group
        zp_ptrs = zero_point_ptr + offs_n[None, :] * stride_zp_row + group_idx[:, None] * stride_zp_group
        scale = tl.load(scale_ptrs)
        zp = tl.load(zp_ptrs)

        # 11. Dequantize
        deq_w_block = (q_vals - zp.to(tl.float32)) * scale
        # deq_w_block = deq_w_block.to(tl.float16)

        # 12. Matrix multiplication
        acc += tl.dot(x_block, deq_w_block)

    # 13. Add bias if needed
    if BIAS_ENABLED:
        bias_ptrs = bias_ptr + offs_n * stride_bias
        bias = tl.load(bias_ptrs, mask=(offs_n < N), other=0.0)
        acc += bias[None, :]

    # 14. Store result
    out_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float16), mask=mask)