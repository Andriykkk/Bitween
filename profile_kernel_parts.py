"""
Profile each part of the CUDA quantized kernel to find bottlenecks.

This script isolates different parts of the kernel:
1. Load X into shared memory
2. Load packed weights
3. Extract bits from packed weights
4. Load scale/zero_point
5. Full dequantization
6. FP16 tensor core matmul (baseline)
"""

import torch
import time
import os
from pathlib import Path
from torch.utils.cpp_extension import load

def load_profiling_kernel():
    """Compile and load the profiling CUDA kernel."""
    kernel_dir = Path(__file__).parent / "bitween" / "kernels"

    print("Compiling profiling CUDA kernels...")

    profiling_module = load(
        name='quantized_matmul_profiling',
        sources=[
            str(kernel_dir / 'cuda_profiling_binding.cpp'),
            str(kernel_dir / 'quantized_matmul_profiling.cu'),
        ],
        extra_cuda_cflags=[
            '-O3',
            '--use_fast_math',
            '-std=c++17',
            '--expt-relaxed-constexpr',
        ],
        verbose=True,
    )

    print("✓ Profiling kernels loaded")
    return profiling_module


def benchmark_kernel(func, warmup=10, iterations=100):
    """Benchmark a kernel function."""
    # Warmup
    for _ in range(warmup):
        func()
        torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iterations):
        func()

    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time_ms = (end - start) / iterations * 1000
    return avg_time_ms


def create_test_data(M, N, K, bits=8, group_size=128, device='cuda'):
    """Create test data for profiling."""
    # Input activation
    x = torch.randn(M, K, dtype=torch.float16, device=device)

    # Quantized weights (packed into int32)
    values_per_int32 = 32 // bits
    packed_K = K // values_per_int32
    qweight = torch.randint(0, 2**31, (N, packed_K), dtype=torch.int32, device=device)

    # Scale and zero point
    num_groups = (K + group_size - 1) // group_size
    scale = torch.randn(N, num_groups, dtype=torch.float16, device=device)
    zero_point = torch.randn(N, num_groups, dtype=torch.float16, device=device)

    # Output
    out = torch.zeros(M, N, dtype=torch.float16, device=device)

    # FP16 weight for matmul baseline
    w_fp16 = torch.randn(N, K, dtype=torch.float16, device=device)

    return x, qweight, scale, zero_point, out, w_fp16


def profile_all_parts():
    """Profile each part of the kernel."""

    # Test configuration
    M, N, K = 512, 512, 512
    bits = 8
    group_size = 128
    device = 'cuda'

    print("\n" + "="*80)
    print(f"CUDA KERNEL PROFILING - Finding Bottlenecks")
    print("="*80)
    print(f"Matrix size: M={M}, N={N}, K={K}")
    print(f"Quantization: {bits}-bit, group_size={group_size}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print("="*80 + "\n")

    # Compile and load kernel
    try:
        prof_module = load_profiling_kernel()
    except Exception as e:
        print(f"Failed to compile/load kernel: {e}")
        return

    # Create test data
    x, qweight, scale, zero_point, out, w_fp16 = create_test_data(M, N, K, bits, group_size, device)

    print("Warming up GPU...")
    for _ in range(20):
        torch.matmul(x, w_fp16.t())
        torch.cuda.synchronize()

    print("\n" + "-"*80)

    # Test each part
    results = {}

    # Part 1: Load X only
    print("\n1. Load X into shared memory (no computation)")
    print("   Testing memory bandwidth for input loading...")
    try:
        time_ms = benchmark_kernel(
            lambda: prof_module.profile_load_x(x, out)
        )
        results['load_x'] = time_ms
        print(f"   ✓ Time: {time_ms:.3f} ms")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        results['load_x'] = None

    # Part 2: Load packed weights only
    print("\n2. Load packed weights (no dequantization)")
    print("   Testing memory bandwidth for packed weight loading...")
    try:
        time_ms = benchmark_kernel(
            lambda: prof_module.profile_load_packed(qweight, out)
        )
        results['load_packed'] = time_ms
        print(f"   ✓ Time: {time_ms:.3f} ms")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        results['load_packed'] = None

    # Part 3: Extract bits
    print("\n3. Load packed + extract bits (no dequantization)")
    print("   Testing bit extraction overhead...")
    try:
        time_ms = benchmark_kernel(
            lambda: prof_module.profile_extract_bits(qweight, out)
        )
        results['extract_bits'] = time_ms
        print(f"   ✓ Time: {time_ms:.3f} ms")
        if results['load_packed']:
            overhead = time_ms - results['load_packed']
            print(f"   → Bit extraction overhead: {overhead:.3f} ms")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        results['extract_bits'] = None

    # Part 4: Load scale/zero_point
    print("\n4. Load scale/zero_point only")
    print("   Testing scale/zp memory access overhead...")
    try:
        time_ms = benchmark_kernel(
            lambda: prof_module.profile_load_scale_zp(scale, zero_point, out, group_size)
        )
        results['load_scale_zp'] = time_ms
        print(f"   ✓ Time: {time_ms:.3f} ms")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        results['load_scale_zp'] = None

    # Part 5: Full dequantization
    print("\n5. Full dequantization (no tensor core matmul)")
    print("   Testing complete dequantization overhead...")
    try:
        time_ms = benchmark_kernel(
            lambda: prof_module.profile_dequantize(qweight, scale, zero_point, out, group_size)
        )
        results['dequantize'] = time_ms
        print(f"   ✓ Time: {time_ms:.3f} ms")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        results['dequantize'] = None

    # Part 6: FP16 tensor core matmul baseline
    print("\n6. FP16 tensor core matmul (baseline)")
    print("   Testing pure compute performance...")
    try:
        time_ms = benchmark_kernel(
            lambda: prof_module.profile_fp16_matmul(x, w_fp16, out)
        )
        results['fp16_matmul'] = time_ms
        print(f"   ✓ Time: {time_ms:.3f} ms")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        results['fp16_matmul'] = None

    # PyTorch FP16 baseline
    print("\n7. PyTorch FP16 matmul (reference)")
    print("   Testing PyTorch optimized implementation...")
    time_ms = benchmark_kernel(
        lambda: torch.matmul(x, w_fp16.t())
    )
    results['pytorch_fp16'] = time_ms
    print(f"   ✓ Time: {time_ms:.3f} ms")

    # Summary
    print("\n" + "="*80)
    print("PROFILING SUMMARY")
    print("="*80)

    print("\nOperation                              Time (ms)    Overhead")
    print("-"*80)

    for name, time_ms in results.items():
        if time_ms is None:
            print(f"{name:40s} FAILED")
        else:
            print(f"{name:40s} {time_ms:8.3f}")

    print("\n" + "="*80)
    print("BOTTLENECK ANALYSIS")
    print("="*80)

    if all(v is not None for v in results.values()):
        total_dequant = results['dequantize']
        fp16_compute = results['fp16_matmul']

        print(f"\nDequantization overhead: {total_dequant:.3f} ms")
        print(f"FP16 tensor core compute: {fp16_compute:.3f} ms")
        print(f"PyTorch FP16 baseline: {results['pytorch_fp16']:.3f} ms")

        print(f"\nBreakdown of dequantization:")
        print(f"  - Load X:              {results['load_x']:.3f} ms ({results['load_x']/total_dequant*100:.1f}%)")
        print(f"  - Load packed weights: {results['load_packed']:.3f} ms ({results['load_packed']/total_dequant*100:.1f}%)")
        print(f"  - Bit extraction:      {results['extract_bits'] - results['load_packed']:.3f} ms ({(results['extract_bits'] - results['load_packed'])/total_dequant*100:.1f}%)")
        print(f"  - Load scale/zp:       {results['load_scale_zp']:.3f} ms ({results['load_scale_zp']/total_dequant*100:.1f}%)")

        # Find the bottleneck
        components = {
            'Load X': results['load_x'],
            'Load packed weights': results['load_packed'],
            'Bit extraction': results['extract_bits'] - results['load_packed'],
            'Load scale/zp': results['load_scale_zp'],
            'Arithmetic operations': total_dequant - (results['load_x'] + results['extract_bits'] + results['load_scale_zp'])
        }

        bottleneck = max(components, key=components.get)
        print(f"\n⚠️  PRIMARY BOTTLENECK: {bottleneck} ({components[bottleneck]:.3f} ms)")

        if total_dequant > fp16_compute:
            slowdown = total_dequant / fp16_compute
            print(f"\n⚠️  Dequantization is {slowdown:.1f}x SLOWER than FP16 compute")
            print("    This explains why quantized kernel is slower than FP16!")

        print("\nRECOMMENDATIONS:")
        if components['Load scale/zp'] > 0.5:
            print("  1. Cache scale/zero_point in shared memory at block level")
            print("  2. Reduce redundant scale/zp loads (same group accessed multiple times)")
        if components['Bit extraction'] > 0.5:
            print("  3. Use vectorized bit operations (load multiple packed ints at once)")
        if components['Load packed weights'] > 0.5:
            print("  4. Optimize memory access pattern for packed weights")
        if total_dequant > fp16_compute * 2:
            print("  5. Consider dequantizing to shared memory ONCE per block")
            print("  6. Use larger tiles to amortize dequantization overhead")


if __name__ == "__main__":
    profile_all_parts()
