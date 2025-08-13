import torch
import torch.nn as nn
import time
import gc
import copy
from transformers import AutoModelForCausalLM
from bitween.utils.singlora import apply_singlora_to_model


def find_all_linear_names(model):
    """Finds all linear layer names to be targeted by LoRA."""
    linear_module_names = {name for name, module in model.named_modules() if isinstance(module, nn.Linear)}
    linear_module_names.discard("lm_head")
    return list(linear_module_names)


def bytes_to_mb(b):
    """Converts bytes to megabytes."""
    return b / (1024**2)


def get_gpu_baseline(device):
    """Get current GPU memory usage in MB."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        return bytes_to_mb(torch.cuda.memory_allocated(device))
    return 0


def measure_peak_memory(model, dummy_input, device, is_training=False):
    """
    Measures peak memory usage during a forward or training pass,
    subtracting baseline GPU usage.
    """
    baseline_mem = get_gpu_baseline(device)

    # Work on a copy of the model to avoid interference
    local_model = copy.deepcopy(model).to(device)

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    if is_training:
        local_model.train()
        optimizer = torch.optim.AdamW(local_model.parameters(), lr=1e-5)
        optimizer.zero_grad()
        output = local_model(**dummy_input, labels=dummy_input["input_ids"])
        loss = output.loss
        loss.backward()
        optimizer.step()
    else:
        local_model.eval()
        with torch.no_grad():
            _ = local_model(**dummy_input)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        peak_memory = bytes_to_mb(torch.cuda.max_memory_allocated(device)) - baseline_mem
    else:
        peak_memory = 0  # CPU version not implemented

    # Cleanup
    del local_model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return peak_memory


def profile_pass(model, dummy_input, device, is_training=False, warmup_runs=5, test_runs=20):
    """Profiles throughput of a forward or training pass without memory interference."""
    local_model = copy.deepcopy(model).to(device)

    if is_training:
        local_model.train()
        optimizer = torch.optim.AdamW(local_model.parameters(), lr=1e-5)
    else:
        local_model.eval()

    # Warmup
    for _ in range(warmup_runs):
        if is_training:
            optimizer.zero_grad()
            output = local_model(**dummy_input, labels=dummy_input["input_ids"])
            loss = output.loss
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                _ = local_model(**dummy_input)

    # Timed runs
    total_time = 0
    for _ in range(test_runs):
        if device.startswith("cuda"): torch.cuda.synchronize()
        start_time = time.perf_counter()

        if is_training:
            optimizer.zero_grad()
            output = local_model(**dummy_input, labels=dummy_input["input_ids"])
            loss = output.loss
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                _ = local_model(**dummy_input)

        if device.startswith("cuda"): torch.cuda.synchronize()
        total_time += (time.perf_counter() - start_time)

    throughput = dummy_input["input_ids"].shape[0] / (total_time / test_runs)

    # Cleanup
    del local_model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return throughput


def generate_lora_benchmark_report(model_path, dummy_input, device, text=""):
    """Generates an isolated benchmark report for a model with LoRA applied."""
    print("\n--- Generating LoRA Fine-Tuning Benchmark Report ---")

    # Load model fresh each time
    model = torch.load(model_path, weights_only=False)

    # Apply LoRA
    target_modules = find_all_linear_names(model)
    apply_singlora_to_model(
        model,
        rank=8,
        alpha=16,
        ramp_up_steps=10,
        target_modules=target_modules,
        print_summary=False
    )
    print(f"Applied SingLoRA to {len(target_modules)} linear layers of " + text + " model.")

    # Benchmark Inference
    print("\n1. Benchmarking Forward Pass (Inference)...")
    fwd_peak_mem = measure_peak_memory(model, dummy_input, device, is_training=False)
    fwd_throughput = profile_pass(model, dummy_input, device, is_training=False)
    print(f"  - Peak Memory (adjusted): {fwd_peak_mem:.2f} MB")
    print(f"  - Throughput: {fwd_throughput:.2f} samples/sec")

    # Benchmark Training
    print("\n2. Benchmarking Training Pass (Forward + Backward)...")
    train_peak_mem = measure_peak_memory(model, dummy_input, device, is_training=True)
    train_throughput = profile_pass(model, dummy_input, device, is_training=True)
    print(f"  - Peak Memory (adjusted): {train_peak_mem:.2f} MB")
    print(f"  - Throughput: {train_throughput:.2f} samples/sec")

    print("\n--- Summary ---")
    print(f"Forward Inference Memory: {fwd_peak_mem:.2f} MB")
    print(f"Training Memory: {train_peak_mem:.2f} MB")
    print(f"Forward Inference Throughput: {fwd_throughput:.2f} samples/sec")
    print(f"Training Throughput: {train_throughput:.2f} samples/sec")
    print("\n--- End of Report ---")
