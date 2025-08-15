import triton
import triton.language as tl

@triton.jit
def quantized_linear_kernel(
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