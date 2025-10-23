"""
Profile CUDA kernel by testing each part separately.
"""

import torch
import time
from pathlib import Path
from torch.utils.cpp_extension import load

def benchmark(func, warmup=10, iterations=100):
    """Benchmark a function."""
    for _ in range(warmup):
        func()
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    torch.cuda.synchronize()

    return (time.perf_counter() - start) / iterations * 1000

def main():
    device = 'cuda'
    M, N, K = 512, 512, 512
    bits = 8
    group_size = 128

    print("\n" + "="*80)
    print("CUDA KERNEL PROFILING - Finding Bottlenecks")
    print("="*80)
    print(f"Matrix size: M={M}, N={N}, K={K}")
    print(f"Quantization: {bits}-bit, group_size={group_size}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print("="*80 + "\n")

    # Compile profiling kernels
    print("Compiling profiling kernels...")
    kernel_dir = Path(__file__).parent / "bitween" / "kernels"
    prof_module = load(
        name='kernel_profile',
        sources=[
            str(kernel_dir / 'profile_binding.cpp'),
            str(kernel_dir / 'quantized_matmul_profile.cu'),
        ],
        extra_cuda_cflags=['-O3', '--use_fast_math', '-std=c++17'],
        verbose=False,
    )
    print("✓ Profiling kernels loaded\n")

    # Create test data
    x = torch.randn(M, K, dtype=torch.float16, device=device)

    values_per_int32 = 32 // bits
    packed_K = K // values_per_int32
    qweight = torch.randint(0, 2**31, (N, packed_K), dtype=torch.int32, device=device)

    num_groups = (K + group_size - 1) // group_size
    scale = torch.randn(N, num_groups, dtype=torch.float16, device=device)
    zero_point = torch.randn(N, num_groups, dtype=torch.float16, device=device)

    out = torch.zeros(M, N, dtype=torch.float16, device=device)

    # Warmup
    print("Warming up GPU...")
    for _ in range(20):
        torch.matmul(x, x.t())
        torch.cuda.synchronize()

    results = {}

    # Test 1: Load X only
    print("\n1. Load X into shared memory (no computation)")
    time_ms = benchmark(lambda: prof_module.profile_load_x(x, out))
    results['load_x'] = time_ms
    print(f"   Time: {time_ms:.3f} ms")

    # Test 2: Load packed weights
    print("\n2. Load packed int32 weights")
    time_ms = benchmark(lambda: prof_module.profile_load_packed(qweight, out))
    results['load_packed'] = time_ms
    print(f"   Time: {time_ms:.3f} ms")

    # Test 3: Load + extract bits
    print("\n3. Load packed + extract bits (shift & mask)")
    time_ms = benchmark(lambda: prof_module.profile_extract_bits(qweight, out))
    results['extract_bits'] = time_ms
    print(f"   Time: {time_ms:.3f} ms")
    print(f"   → Bit extraction overhead: {results['extract_bits'] - results['load_packed']:.3f} ms")

    # Test 4: Load scale/zero_point
    print("\n4. Load scale/zero_point only")
    time_ms = benchmark(lambda: prof_module.profile_load_scale_zp(scale, zero_point, out, group_size))
    results['load_scale_zp'] = time_ms
    print(f"   Time: {time_ms:.3f} ms")

    # Test 5: Full dequantization (no WMMA)
    print("\n5. Full dequantization (no tensor cores)")
    time_ms = benchmark(lambda: prof_module.profile_dequant(qweight, scale, zero_point, out, group_size))
    results['dequant'] = time_ms
    print(f"   Time: {time_ms:.3f} ms")

    # Test 6: Full kernel (dequant + WMMA)
    print("\n6. Full kernel (dequant + WMMA)")
    time_ms = benchmark(lambda: prof_module.profile_full_kernel(x, qweight, scale, zero_point, out, group_size))
    results['full_kernel'] = time_ms
    print(f"   Time: {time_ms:.3f} ms")
    if results['dequant']:
        wmma_time = results['full_kernel'] - results['dequant']
        print(f"   → WMMA overhead: {wmma_time:.3f} ms")

    # PyTorch baseline
    print("\n7. PyTorch FP16 matmul (reference)")
    w_fp16 = torch.randn(N, K, dtype=torch.float16, device=device)
    time_ms = benchmark(lambda: torch.matmul(x, w_fp16.t()))
    results['pytorch'] = time_ms
    print(f"   Time: {time_ms:.3f} ms")

    # Analysis
    print("\n" + "="*80)
    print("BOTTLENECK ANALYSIS")
    print("="*80)

    print(f"\nTime breakdown:")
    print(f"  Load X:              {results['load_x']:.3f} ms ({results['load_x']/results['full_kernel']*100:.1f}%)")
    print(f"  Load packed:         {results['load_packed']:.3f} ms ({results['load_packed']/results['full_kernel']*100:.1f}%)")
    print(f"  Bit extraction:      {results['extract_bits'] - results['load_packed']:.3f} ms ({(results['extract_bits'] - results['load_packed'])/results['full_kernel']*100:.1f}%)")
    print(f"  Load scale/zp:       {results['load_scale_zp']:.3f} ms ({results['load_scale_zp']/results['full_kernel']*100:.1f}%)")
    print(f"  Full dequant:        {results['dequant']:.3f} ms ({results['dequant']/results['full_kernel']*100:.1f}%)")
    print(f"  WMMA:                {results['full_kernel'] - results['dequant']:.3f} ms ({(results['full_kernel'] - results['dequant'])/results['full_kernel']*100:.1f}%)")
    print(f"  ---")
    print(f"  Full kernel:         {results['full_kernel']:.3f} ms")
    print(f"  PyTorch FP16:        {results['pytorch']:.3f} ms")

    # Find bottleneck
    components = {
        'Load X': results['load_x'],
        'Load packed': results['load_packed'],
        'Bit extraction': results['extract_bits'] - results['load_packed'],
        'Load scale/zp': results['load_scale_zp'],
        'Dequant arithmetic': results['dequant'] - results['load_x'] - results['extract_bits'] - results['load_scale_zp'],
        'WMMA': results['full_kernel'] - results['dequant'],
    }

    bottleneck = max(components, key=components.get)
    print(f"\n⚠️  PRIMARY BOTTLENECK: {bottleneck} ({components[bottleneck]:.3f} ms, {components[bottleneck]/results['full_kernel']*100:.1f}%)")

    slowdown = results['full_kernel'] / results['pytorch']
    print(f"\n⚠️  Quantized kernel is {slowdown:.1f}x SLOWER than PyTorch FP16")

    print("\nRECOMMENDATIONS:")
    if components['Load scale/zp'] > 0.1:
        print("  1. ⚠️  Scale/ZP loading is slow - cache in shared memory")
    if components['Bit extraction'] > 0.05:
        print("  2. Bit extraction overhead - consider vectorized operations")
    if components['WMMA'] < 0.01:
        print("  3. ✓ WMMA is fast - tensor cores working well")
    if results['dequant'] > results['pytorch'] * 5:
        print("  4. ⚠️  Dequantization dominates - consider larger tiles to amortize cost")

if __name__ == "__main__":
    main()
