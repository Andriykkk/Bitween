import torch
import torch.nn as nn
import time
import copy
import os
import gc
from bitween import Bitween
from torch.profiler import profile, record_function, ProfilerActivity

# --- Test Model ---
class SimpleMLP(nn.Module):
    """A simple Multi-Layer Perceptron for benchmarking."""
    def __init__(self, input_dim=512, hidden_dim=1024, output_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


# --- Utility Functions ---
def bytes_to_mb(b):
    """Converts bytes to megabytes."""
    return b / (1024**2)

def get_gpu_baseline(device):
    """Get current GPU memory usage in MB."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        return bytes_to_mb(torch.cuda.memory_allocated(device))
    return 0

def measure_peak_memory(model, dummy_input, device):
    """Measures peak GPU memory usage for one forward pass."""
    baseline_mem = get_gpu_baseline(device)
    local_model = copy.deepcopy(model).to(device)

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    local_model.eval()
    with torch.no_grad():
        _ = local_model(dummy_input)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        peak_memory = bytes_to_mb(torch.cuda.max_memory_allocated(device)) - baseline_mem
    else:
        peak_memory = 0  # CPU case

    del local_model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return peak_memory

def profile_speed(model, input_tensor, device, warmup_runs=10, test_runs=50):
    """Profiles inference throughput."""
    local_model = copy.deepcopy(model).to(device)
    local_model.eval()

    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = local_model(input_tensor)

    # Timed runs
    total_time = 0
    with torch.no_grad():
        for _ in range(test_runs):
            if device.startswith("cuda"): torch.cuda.synchronize()
            start_time = time.perf_counter()
            _ = local_model(input_tensor)
            if device.startswith("cuda"): torch.cuda.synchronize()
            total_time += (time.perf_counter() - start_time)

    throughput = input_tensor.shape[0] / (total_time / test_runs)

    del local_model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return throughput

def get_gpu_baseline(device):
    """Get current GPU memory usage in MB."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        return bytes_to_mb(torch.cuda.memory_allocated(device))
    return 0

def measure_peak_memory(model, dummy_input, device):
    """Measures peak GPU memory usage for one forward pass."""
    baseline_mem = get_gpu_baseline(device)
    local_model = copy.deepcopy(model).to(device)

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    local_model.eval()
    with torch.no_grad():
        _ = local_model(dummy_input)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        peak_memory = bytes_to_mb(torch.cuda.max_memory_allocated(device)) - baseline_mem
    else:
        peak_memory = 0  # CPU case

    del local_model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return peak_memory

def profile_speed(model, input_tensor, device, warmup_runs=10, test_runs=50):
    """Profiles inference throughput."""
    local_model = copy.deepcopy(model).to(device)
    local_model.eval()

    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = local_model(input_tensor)

    # Timed runs
    total_time = 0
    with torch.no_grad():
        for _ in range(test_runs):
            if device.startswith("cuda"): torch.cuda.synchronize()
            start_time = time.perf_counter()
            _ = local_model(input_tensor)
            if device.startswith("cuda"): torch.cuda.synchronize()
            total_time += (time.perf_counter() - start_time)

    throughput = input_tensor.shape[0] / (total_time / test_runs)

    del local_model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return throughput

def measure_peak_memory(model, dummy_input, device):
    """Measures peak GPU memory usage for one forward pass."""
    baseline_mem = get_gpu_baseline(device)
    local_model = copy.deepcopy(model).to(device)

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    local_model.eval()
    with torch.no_grad():
        _ = local_model(dummy_input)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        peak_memory = bytes_to_mb(torch.cuda.max_memory_allocated(device)) - baseline_mem
    else:
        peak_memory = 0  # CPU case

    del local_model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return peak_memory

def measure_memory(model):
    """Measures the memory footprint of model parameters."""
    param_memory = 0
    for param in model.parameters():
        param_memory += param.numel() * param.element_size()
    return bytes_to_mb(param_memory)

def profile_speed(model, input_tensor, device, warmup_runs=10, test_runs=20):
    """Profiles inference throughput."""
    local_model = copy.deepcopy(model).to(device)
    local_model.eval()

    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = local_model(input_tensor)

    # Timed runs
    total_time = 0
    with torch.no_grad():
        for _ in range(test_runs):
            if device.startswith("cuda"): torch.cuda.synchronize()
            start_time = time.perf_counter()
            _ = local_model(input_tensor)
            if device.startswith("cuda"): torch.cuda.synchronize()
            total_time += (time.perf_counter() - start_time)

    throughput = input_tensor.shape[0] / (total_time / test_runs)

    del local_model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return throughput

def generate_benchmark_report(fp32_model_path, quantized_model_path, input_tensor, device):
    """Loads models and benchmarks with real GPU memory usage."""
    print("\n--- Generating Benchmark Report ---")

    # --- FP32 Model ---
    print("1. Benchmarking FP32 Model...")
    fp32_model = torch.load(fp32_model_path, weights_only=False)
    fp32_memory = measure_peak_memory(fp32_model, input_tensor, device)
    fp32_throughput = profile_speed(fp32_model, input_tensor, device)
    print(f" - Peak Memory: {fp32_memory:.2f} MB")
    print(f" - Throughput: {fp32_throughput:.2f} samples/sec")
    del fp32_model
    gc.collect()
    if device.startswith("cuda"): torch.cuda.empty_cache()

    # --- Quantized Model ---
    print("\n2. Benchmarking Quantized Model...")
    quantized_model = torch.load(quantized_model_path, weights_only=False)
    quantized_memory = measure_peak_memory(quantized_model, input_tensor, device)
    quantized_throughput = profile_speed(quantized_model, input_tensor, device)
    print(f" - Peak Memory: {quantized_memory:.2f} MB")
    print(f" - Throughput: {quantized_throughput:.2f} samples/sec")
    del quantized_model
    gc.collect()
    if device.startswith("cuda"): torch.cuda.empty_cache()

    # --- Summary ---
    print("\n--- 3. Comparison Summary ---")
    mem_reduction = (1 - (quantized_memory / fp32_memory)) * 100 if fp32_memory > 0 else 0
    speed_increase = ((quantized_throughput - fp32_throughput) / fp32_throughput) * 100 if fp32_throughput > 0 else 0
    print(f"Memory Reduction: {mem_reduction:.2f}%")
    print(f"Throughput Change: {speed_increase:+.2f}%")
    print("-" * 35)


    """Loads models and benchmarks with real GPU memory usage."""
    print("\n--- Generating Benchmark Report ---")

    # --- FP32 Model ---
    print("1. Benchmarking FP32 Model...")
    fp32_model = torch.load(fp32_model_path, weights_only=False)
    fp32_memory = measure_peak_memory(fp32_model, input_tensor, device)
    fp32_throughput = profile_speed(fp32_model, input_tensor, device)
    print(f" - Peak Memory: {fp32_memory:.2f} MB")
    print(f" - Throughput: {fp32_throughput:.2f} samples/sec")
    del fp32_model
    gc.collect()
    if device.startswith("cuda"): torch.cuda.empty_cache()

    # --- Quantized Model ---
    print("\n2. Benchmarking Quantized Model...")
    quantized_model = torch.load(quantized_model_path, weights_only=False)
    quantized_memory = measure_peak_memory(quantized_model, input_tensor, device)
    quantized_throughput = profile_speed(quantized_model, input_tensor, device)
    print(f" - Peak Memory: {quantized_memory:.2f} MB")
    print(f" - Throughput: {quantized_throughput:.2f} samples/sec")
    del quantized_model
    gc.collect()
    if device.startswith("cuda"): torch.cuda.empty_cache()

    # --- Summary ---
    print("\n--- 3. Comparison Summary ---")
    mem_reduction = (1 - (quantized_memory / fp32_memory)) * 100 if fp32_memory > 0 else 0
    speed_increase = ((quantized_throughput - fp32_throughput) / fp32_throughput) * 100 if fp32_throughput > 0 else 0
    print(f"Memory Reduction: {mem_reduction:.2f}%")
    print(f"Throughput Change: {speed_increase:+.2f}%")
    print("-" * 35)

def generate_benchmark_report(fp32_model_path, quantized_model_path, input_tensor, device):
    """
    Loads models and benchmarks with torch.profiler.
    Replaces the original speed and memory functions.
    """

    """
    Loads models from files and generates a detailed performance report.
    This function can be imported and used in other scripts.
    """
    print("\n--- Generating Benchmark Report ---")
    
    # --- Load and Benchmark FP32 Model ---
    print("1. Benchmarking FP32 Model...")
    fp32_model = torch.load(fp32_model_path, weights_only=False)
    fp32_memory = measure_memory(fp32_model)
    fp32_throughput = profile_speed(fp32_model, input_tensor, device)
    print(f"  - Memory: {fp32_memory:.2f} MB")
    print(f"  - Throughput: {fp32_throughput:.2f} samples/sec")
    del fp32_model # Clean up memory
    if device == 'cuda': torch.cuda.empty_cache()

    # --- Load and Benchmark Quantized Model ---
    print("\n2. Benchmarking Quantized Model...")
    quantized_model = torch.load(quantized_model_path, weights_only=False)
    quantized_memory = measure_memory(quantized_model)
    quantized_throughput = profile_speed(quantized_model, input_tensor, device)
    print(f"  - Memory: {quantized_memory:.2f} MB")
    print(f"  - Throughput: {quantized_throughput:.2f} samples/sec")
    del quantized_model # Clean up memory
    if device == 'cuda': torch.cuda.empty_cache()

    # --- Summary ---
    print("\n--- 3. Comparison Summary ---")
    mem_reduction = (1 - (quantized_memory / fp32_memory)) * 100 if fp32_memory > 0 else 0
    speed_increase = ((quantized_throughput - fp32_throughput) / fp32_throughput) * 100 if fp32_throughput > 0 else 0
    
    print(f"Memory Reduction: {mem_reduction:.2f}%")
    print(f"Throughput Change: {speed_increase:+.2f}%")
    print("-" * 35)

    print("\n--- Generating Benchmark Report with torch.profiler ---")

    # --- FP32 Model ---
    print("1. Profiling FP32 Model...")
    fp32_model = torch.load(fp32_model_path, weights_only=False).to(device).eval()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof_fp32:
        with record_function("FP32_MLP_Forward"):
            with torch.no_grad():
                _ = fp32_model(input_tensor)
    print(prof_fp32.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    del fp32_model
    gc.collect()
    if device.startswith("cuda"): torch.cuda.empty_cache()

    # --- Quantized Model ---
    print("\n2. Profiling Quantized Model...")
    quantized_model = torch.load(quantized_model_path, weights_only=False).to(device).eval()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof_quant:
        with record_function("Quantized_MLP_Forward"):
            with torch.no_grad():
                _ = quantized_model(input_tensor)
    print(prof_quant.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    del quantized_model
    gc.collect()
    if device.startswith("cuda"): torch.cuda.empty_cache()

    # --- Sanity check and comparison summary ---
    print("\n--- Comparison Summary ---")
    fp32_model = torch.load(fp32_model_path, weights_only=False).to(device).eval()
    quantized_model = torch.load(quantized_model_path, weights_only=False).to(device).eval()
    with torch.no_grad():
        fp32_output = fp32_model(input_tensor)
        quant_output = quantized_model(input_tensor)
    
    # Extract logits if outputs are objects with .logits attribute
    if hasattr(fp32_output, 'logits'):
        fp32_output = fp32_output.logits
    if hasattr(quant_output, 'logits'):
        quant_output = quant_output.logits
        
    max_error = (fp32_output - quant_output).abs().max()
    print(f"Max Error between FP32 and Quantized outputs: {max_error:.4f}")
    del fp32_model, quantized_model
    gc.collect()
    if device.startswith("cuda"): torch.cuda.empty_cache()
    print("-" * 35)


if __name__ == "__main__":
    # --- Configuration ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_dim = 1024
    batch_size = 32
    save_dir = "temp_models"
    os.makedirs(save_dir, exist_ok=True)

    fp32_path = os.path.join(save_dir, "fp32_model.pth")
    quantized_path = os.path.join(save_dir, "quantized_model.pth")

    print(f"Running on device: {device}")
    print(f"Models will be saved to '{save_dir}' directory.")
    print("-" * 35)

    # --- Create & Save FP32 Model ---
    print("--- Creating and saving original FP32 model ---")
    fp32_model = SimpleMLP(input_dim=input_dim).to(device)
    torch.save(fp32_model, fp32_path)

    # --- Quantize & Save Model ---
    print("\n--- Quantizing and saving model ---")
    model_to_quantize = copy.deepcopy(fp32_model)
    quantizer = Bitween(model_to_quantize, bits=8, group_size=-1)
    quantized_model = quantizer.quantize(evaluate_perplexity=False)
    torch.save(quantized_model, quantized_path)

    # --- Run Benchmark ---
    input_tensor = torch.randn(batch_size, input_dim).to(device)
    generate_benchmark_report(fp32_path, quantized_path, input_tensor, device)
