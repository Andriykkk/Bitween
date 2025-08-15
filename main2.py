import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Set up the device
device = 'cuda'
torch.cuda.manual_seed(123)

# --- Provided quantization and dequantization functions ---
# Note: These are taken from your provided code to prepare the weights.
def _quantize_8bit(tensor, scale, maxq):
    q_tensor = (tensor / scale).round().clamp(-maxq, maxq)
    q_tensor = (q_tensor + maxq).to(torch.uint8)
    return q_tensor

def _quantize_4bit(tensor, scale, maxq):
    # This function is not used in the kernel but is kept for completeness.
    q_tensor = (tensor / scale).round().clamp(-maxq, maxq)
    q_tensor = (q_tensor + maxq).to(torch.uint8)
    packed_tensor = q_tensor[:, :, ::2] | (q_tensor[:, :, 1::2] << 4)
    return packed_tensor

def quantize_rtn(tensor, bits=8, group_size=-1, eps=1e-5):
    assert bits in [4, 8], "Only 4-bit and 8-bit quantization are supported."
    if bits == 4 and tensor.shape[-1] % 2 != 0:
        raise ValueError("For 4-bit quantization, the last dimension must be even.")

    maxq = 2 ** (bits - 1) - 1
    shape = tensor.shape
    if group_size != -1:
        if shape[-1] % group_size != 0:
            raise ValueError("The last dimension must be divisible by group_size.")
        t_view = tensor.view(shape[0], -1, group_size)
    else:
        t_view = tensor.view(shape[0], 1, shape[-1])
    max_val = t_view.abs().max(dim=2, keepdim=True)[0]
    scale = (max_val / maxq).clamp(min=eps)

    if bits == 8:
        q_view = _quantize_8bit(t_view, scale, maxq)
        q_tensor = q_view.view(*shape)
    else: # bits == 4
        q_view = _quantize_4bit(t_view, scale, maxq)
        q_tensor = q_view.view(shape[0], -1)

    zero_point = torch.full_like(scale, maxq)
    if group_size == -1:
        scale = scale.squeeze(1)
        zero_point = zero_point.squeeze(1)

    return q_tensor, scale, zero_point, group_size


# --- Triton kernel for quantized linear forward pass ---
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

# --- Wrapper function to launch the kernel ---
def triton_quantized_linear(x, qweight, scale, zero_point, bias=None):
    # Ensure all tensors are on the GPU
    x = x.to(device)
    qweight = qweight.to(device)
    scale = scale.to(device)
    zero_point = zero_point.to(device)
    if bias is not None:
        bias = bias.to(device)

    # Get dimensions
    M, K = x.shape
    N, _ = qweight.shape

    # Allocate the output tensor
    c = torch.empty((M, N), device=device, dtype=torch.float32)

    # Block sizes and grid dimensions
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    # Launch the kernel
    quantized_linear_kernel[grid](
        x, qweight, scale, zero_point, bias, c,
        M, N, K,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        scale.stride(0),
        zero_point.stride(0),
        bias.stride(0) if bias is not None else 0,
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BIAS_ENABLED=(bias is not None)
    )
    
    return c


# --- Main execution ---
if __name__ == '__main__':
    # Define layer dimensions
    in_features, out_features = 512, 1024
    
    # Instantiate a float linear layer
    float_layer = nn.Linear(in_features, out_features, bias=True).to(device)
    
    # Quantize the weights using the provided function (per-row)
    qweight, scale, zero_point, _ = quantize_rtn(float_layer.weight.data, bits=8, group_size=-1)
    
    # Create a random input tensor
    x = torch.randn(256, in_features, device=device, dtype=torch.float16)

    # 1. Forward pass using the Triton kernel
    triton_output = triton_quantized_linear(x, qweight, scale, zero_point, float_layer.bias)

    # 2. Reference forward pass (PyTorch's F.linear)
    # First, dequantize the weights on the CPU for the reference calculation.
    # Note that `zero_point` is the value `maxq`, which is a constant for 8-bit symmetric quantization.
    maxq = 2**(8-1)-1
    ref_weight = (qweight.float() - maxq) * scale.view(-1, 1)
    
    # Then, perform the linear operation using the dequantized weights.
    # We use half precision for the reference to match the Triton kernel's input.
    pytorch_output = F.linear(x, ref_weight.half(), float_layer.bias.half())
    
    # 3. Verify the results
    torch.testing.assert_close(triton_output.float(), pytorch_output.float(), rtol=1e-2, atol=1e-2)
    print("Verification successful: Triton quantized kernel output matches PyTorch's output.")
    print("max error:", (triton_output.float() - pytorch_output.float()).abs().max())
    print("max error2:", (triton_output.float() - (float_layer(x.float()).float())).abs().max())