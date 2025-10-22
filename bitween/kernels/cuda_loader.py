import os
import torch
from torch.utils.cpp_extension import load

# Get the directory where this file is located
kernel_dir = os.path.dirname(os.path.abspath(__file__))

# Compile and load the CUDA extension
# This happens once on first import, then cached
cuda_module = load(
    name='quantized_matmul_cuda',
    sources=[
        os.path.join(kernel_dir, 'cuda_binding.cpp'),
        os.path.join(kernel_dir, 'quantized_matmul.cu'),
    ],
    extra_cuda_cflags=[
        '-O3',
        '--use_fast_math',
        '-std=c++17',
        '--expt-relaxed-constexpr',
    ],
    verbose=True,
)

def quantized_matmul_cuda(x, qweight, scale, zero_point, bias, bits, group_size):
    """
    Optimized CUDA kernel for quantized matrix multiplication.

    Args:
        x: Input tensor [M, K] or [..., M, K] (float16)
        qweight: Packed quantized weights [N, packed_K] (int32)
        scale: Per-group scales [N, num_groups] (float16)
        zero_point: Per-group zero points [N, num_groups] (float16)
        bias: Bias [N] (float16) or None
        bits: Number of bits (2, 4, or 8)
        group_size: Group size for quantization

    Returns:
        Output tensor [..., M, N] (float16)
    """
    return cuda_module.quantized_matmul_forward(
        x, qweight, scale, zero_point, bias, bits, group_size
    )
