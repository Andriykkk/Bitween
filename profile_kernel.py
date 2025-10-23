"""
Profile and compare four approaches:
1. PyTorch FP16 matmul (baseline)
2. Custom CUDA FP16 matmul kernel (WMMA)
3. Triton quantized INT8 kernel
4. CUDA quantized INT8 kernel (optimized)

This helps identify performance characteristics and compare Triton vs CUDA implementations.
"""

import torch
import torch.nn as nn
from bitween import QuantizedLinear
import triton
import triton.language as tl
from pathlib import Path
from torch.utils.cpp_extension import load


# Simple Triton matmul kernel for comparison
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Standard Triton matmul kernel with grouped ordering: C = A @ B
    A is (M, K), B is (K, N), C is (M, N)
    Uses grouped ordering for better L2 cache locality.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Grouped ordering for L2 cache locality
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Offsets for this block
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to blocks
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load blocks
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)

        # Matrix multiply using tensor cores
        accumulator += tl.dot(a, b)

        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Convert to output dtype and write
    c = accumulator.to(tl.float16)

    # Output pointers
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def cuda_fp16_matmul(a, b, fp16_module):
    """Wrapper for custom CUDA FP16 WMMA matmul"""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    return fp16_module.fp16_matmul_wmma(a, b)


def profile_all():
    print("="*70)
    print("MATMUL PERFORMANCE COMPARISON")
    print("="*70)

    size = 4096  # Larger matrix for more accurate benchmarking
    batch = 16   # Larger batch to better utilize GPU
    bits = 8
    group_size = 128

    print(f"\nMatrix size: {size}x{size}")
    print(f"Batch size: {batch}")
    print(f"Quantization: {bits}-bit, group_size={group_size}")

    # Create test data - use FP16 for fair comparison
    torch.manual_seed(42)
    float_layer_fp32 = nn.Linear(size, size, bias=True, dtype=torch.float32).cuda()
    float_layer = nn.Linear(size, size, bias=True, dtype=torch.float16).cuda()
    float_layer.weight.data.copy_(float_layer_fp32.weight.data.to(torch.float16))
    if float_layer_fp32.bias is not None:
        float_layer.bias.data.copy_(float_layer_fp32.bias.data.to(torch.float16))

    x = torch.randn(batch, size, device='cuda', dtype=torch.float16)

    # Prepare for custom matmul test
    weight = float_layer.weight.data.t().contiguous()  # (size, size)

    # Prepare quantized layer (from FP32 for better quantization)
    q_layer = QuantizedLinear.from_float(float_layer_fp32, bits=bits, group_size=group_size).cuda()

    # Load CUDA kernels
    try:
        from bitween.kernels.cuda_loader import quantized_matmul_cuda
        cuda_available = True
        print("\n✓ CUDA quantized kernel loaded")
    except Exception as e:
        cuda_available = False
        print(f"\n✗ CUDA quantized kernel not available: {e}")

    # Load CUDA FP16 kernel
    try:
        print("Compiling CUDA FP16 kernel...")
        kernel_dir = Path(__file__).parent / "bitween" / "kernels"

        # Force recompilation by using a unique build directory
        import time
        build_dir = Path(__file__).parent / "build" / f"fp16_build_{int(time.time())}"
        build_dir.mkdir(parents=True, exist_ok=True)

        fp16_module = load(
            name='fp16_matmul_cuda',
            sources=[
                str(kernel_dir / 'fp16_matmul_binding.cpp'),
                str(kernel_dir / 'fp16_matmul.cu'),
            ],
            build_directory=str(build_dir),
            extra_cuda_cflags=['-O3', '--use_fast_math', '-std=c++17'],
            verbose=True,
        )
        cuda_fp16_available = True
        print("✓ CUDA FP16 kernel loaded")
    except Exception as e:
        cuda_fp16_available = False
        fp16_module = None
        print(f"✗ CUDA FP16 kernel not available: {e}")

    # Warm up
    print("\nWarming up...")
    for _ in range(20):
        _ = float_layer(x)
        if cuda_fp16_available:
            _ = cuda_fp16_matmul(x, weight, fp16_module)
        _ = q_layer(x)
        if cuda_available:
            _ = quantized_matmul_cuda(x, q_layer.qweight, q_layer.scale, q_layer.zero_point, q_layer.bias, bits, group_size)
    torch.cuda.synchronize()

    # Extra warm-up for kernels to ensure JIT compilation is complete
    print("Extra warm-up for CUDA/Triton JIT compilation...")
    for _ in range(50):
        if cuda_fp16_available:
            _ = cuda_fp16_matmul(x, weight, fp16_module)
        _ = q_layer(x)
        if cuda_available:
            _ = quantized_matmul_cuda(x, q_layer.qweight, q_layer.scale, q_layer.zero_point, q_layer.bias, bits, group_size)
    torch.cuda.synchronize()

    # Benchmark 1: PyTorch matmul
    print("\n" + "="*70)
    print("1. PyTorch FP16 Matmul (baseline)")
    print("="*70)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    iters = 50
    start.record()
    for _ in range(iters):
        y_pt = float_layer(x)
    end.record()
    torch.cuda.synchronize()
    pytorch_time = start.elapsed_time(end) / iters

    # Calculate memory bandwidth
    weight_bytes = size * size * 2  # fp16
    input_bytes = batch * size * 2
    output_bytes = batch * size * 2
    total_bytes_pt = weight_bytes + input_bytes + output_bytes
    pytorch_bw = (total_bytes_pt / 1e9) / (pytorch_time / 1000)

    print(f"  Time: {pytorch_time:.3f} ms")
    print(f"  Memory: {total_bytes_pt / 1e6:.1f} MB")
    print(f"  Bandwidth: {pytorch_bw:.1f} GB/s")

    # Benchmark 2: Custom CUDA FP16 WMMA matmul
    print("\n" + "="*70)
    print("2. Custom CUDA FP16 Matmul (WMMA)")
    print("="*70)

    if cuda_fp16_available:
        start.record()
        for _ in range(iters):
            y_cuda_fp16 = cuda_fp16_matmul(x, weight, fp16_module)
        end.record()
        torch.cuda.synchronize()
        cuda_fp16_time = start.elapsed_time(end) / iters
    else:
        cuda_fp16_time = None
        print("  SKIPPED - kernel not available")

    if cuda_fp16_available:
        # Calculate memory bandwidth
        weight_bytes_fp16 = size * size * 2  # fp16
        input_bytes_fp16 = batch * size * 2
        output_bytes_fp16 = batch * size * 2
        total_bytes_cuda_fp16 = weight_bytes_fp16 + input_bytes_fp16 + output_bytes_fp16
        cuda_fp16_bw = (total_bytes_cuda_fp16 / 1e9) / (cuda_fp16_time / 1000)

        print(f"  Time: {cuda_fp16_time:.3f} ms")
        print(f"  Memory: {total_bytes_cuda_fp16 / 1e6:.1f} MB")
        print(f"  Bandwidth: {cuda_fp16_bw:.1f} GB/s")
        print(f"  Speedup vs PyTorch: {pytorch_time / cuda_fp16_time:.2f}x")

    # Benchmark 3: Quantized kernel
    print("\n" + "="*70)
    print("3. Quantized INT8 Triton Kernel")
    print("="*70)

    start.record()
    for _ in range(iters):
        y_q = q_layer(x)
    end.record()
    torch.cuda.synchronize()
    quantized_time = start.elapsed_time(end) / iters

    # Calculate quantized memory
    qweight_bytes = size * size * 1  # 8-bit packed
    scale_bytes = size * (size // group_size) * 2  # fp16
    zp_bytes = size * (size // group_size) * 2  # fp16
    total_bytes_q = qweight_bytes + scale_bytes + zp_bytes + input_bytes + output_bytes
    quantized_bw = (total_bytes_q / 1e9) / (quantized_time / 1000)

    print(f"  Time: {quantized_time:.3f} ms")
    print(f"  Memory: {total_bytes_q / 1e6:.1f} MB")
    print(f"  Bandwidth: {quantized_bw:.1f} GB/s")
    print(f"  Speedup vs PyTorch: {pytorch_time / quantized_time:.2f}x")
    if cuda_fp16_available:
        print(f"  Speedup vs CUDA FP16: {cuda_fp16_time / quantized_time:.2f}x")

    # Benchmark 4: CUDA Quantized kernel
    if cuda_available:
        print("\n" + "="*70)
        print("4. CUDA Quantized INT8 Kernel (optimized)")
        print("="*70)

        start.record()
        for _ in range(iters):
            y_cuda = quantized_matmul_cuda(x, q_layer.qweight, q_layer.scale, q_layer.zero_point, q_layer.bias, bits, group_size)
        end.record()
        torch.cuda.synchronize()
        cuda_time = start.elapsed_time(end) / iters

        # Same memory as regular quantized
        cuda_bw = (total_bytes_q / 1e9) / (cuda_time / 1000)

        print(f"  Time: {cuda_time:.3f} ms")
        print(f"  Memory: {total_bytes_q / 1e6:.1f} MB")
        print(f"  Bandwidth: {cuda_bw:.1f} GB/s")
        print(f"  Speedup vs PyTorch: {pytorch_time / cuda_time:.2f}x")
        if cuda_fp16_available:
            print(f"  Speedup vs CUDA FP16: {cuda_fp16_time / cuda_time:.2f}x")
        print(f"  Speedup vs Triton Quantized: {quantized_time / cuda_time:.2f}x")
    else:
        cuda_time = None
        cuda_bw = None

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Method':<30} {'Time (ms)':<12} {'Bandwidth (GB/s)':<18} {'Speedup':<10}")
    print("-"*70)
    print(f"{'PyTorch FP16':<30} {pytorch_time:>10.3f}   {pytorch_bw:>16.1f}   {'1.00x':>8}")
    if cuda_fp16_available:
        print(f"{'CUDA FP16 (WMMA)':<30} {cuda_fp16_time:>10.3f}   {cuda_fp16_bw:>16.1f}   {pytorch_time/cuda_fp16_time:>8.2f}x")
    print(f"{'Triton Quantized INT8':<30} {quantized_time:>10.3f}   {quantized_bw:>16.1f}   {pytorch_time/quantized_time:>8.2f}x")
    if cuda_available:
        print(f"{'CUDA Quantized INT8':<30} {cuda_time:>10.3f}   {cuda_bw:>16.1f}   {pytorch_time/cuda_time:>8.2f}x")

    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    if cuda_available and cuda_fp16_available:
        print(f"Triton vs CUDA Quantized: {quantized_time / cuda_time:.2f}x speedup with CUDA")

        if cuda_time > cuda_fp16_time:
            slowdown = cuda_time / cuda_fp16_time
            print(f"\n⚠️  CUDA quantized kernel is {slowdown:.2f}x SLOWER than CUDA FP16")
            print(f"   This suggests:")
            print(f"     - Dequantization overhead dominates")
            print(f"     - Need to optimize shared memory usage")
            print(f"     - May need tensor cores for matmul")
        else:
            speedup = cuda_fp16_time / cuda_time
            print(f"\n✅ CUDA quantized kernel is {speedup:.2f}x FASTER than CUDA FP16!")
            print(f"   This is the expected behavior - less memory traffic wins!")

        if cuda_bw < cuda_fp16_bw * 0.5:
            print(f"\n⚠️  CUDA quantized bandwidth ({cuda_bw:.1f} GB/s) is lower than")
            print(f"   CUDA FP16 bandwidth ({cuda_fp16_bw:.1f} GB/s)")
            print(f"   But this may be expected - more compute per byte loaded")
        else:
            print(f"\n✅ CUDA quantized bandwidth ({cuda_bw:.1f} GB/s) is comparable to")
            print(f"   CUDA FP16 bandwidth ({cuda_fp16_bw:.1f} GB/s)")
    else:
        print("\n⚠️  CUDA kernels not fully available - skipping analysis")

    # Verify correctness
    print("\n" + "="*70)
    print("CORRECTNESS CHECK")
    print("="*70)

    y_pt = float_layer(x)
    y_q = q_layer(x)

    if cuda_fp16_available:
        y_cuda_fp16 = cuda_fp16_matmul(x, weight, fp16_module)
        error_cuda_fp16 = (y_pt - y_cuda_fp16).abs().max().item()
        print(f"Max error (PyTorch vs CUDA FP16):        {error_cuda_fp16:.6f}")

    error_q = (y_pt - y_q).abs().max().item()
    print(f"Max error (PyTorch vs Triton Quantized): {error_q:.6f}")

    if cuda_available:
        y_cuda = quantized_matmul_cuda(x, q_layer.qweight, q_layer.scale, q_layer.zero_point, q_layer.bias, bits, group_size)
        error_cuda = (y_pt - y_cuda).abs().max().item()
        error_cuda_vs_triton = (y_q - y_cuda).abs().max().item()
        print(f"Max error (PyTorch vs CUDA Quantized):   {error_cuda:.6f}")
        print(f"Max error (Triton vs CUDA Quantized):    {error_cuda_vs_triton:.6f}")

    return pytorch_time, cuda_fp16_time if cuda_fp16_available else None, quantized_time, cuda_time if cuda_available else None


if __name__ == "__main__":
    profile_all()
