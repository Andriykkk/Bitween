import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import time
import matplotlib.pyplot as plt
from model import GPT, GPTConfig
from quantisation.q16bit_partial import convert_to_partial_16bit

# --- Analyzer Settings ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# <<< CHANGE THIS TO TEST DIFFERENT MODELS >>>
QUANTIZATION_MODE = 'bf16_partial' # Options: 'none', 'bf16_partial'
# Test configurations: context sizes to test
context_sizes = [64, 128, 256, 512, 1024]
warmup_runs = 5
test_runs = 10

# --- Model Configurations to Test ---
# Define a dictionary of model configurations to compare
model_configs = {
    "Small (n_embd=384, n_layer=6)": GPTConfig(
        vocab_size=50257, block_size=max(context_sizes), n_embd=384, n_head=6, n_layer=6, dropout=0.1
    ),
    "Medium (n_embd=768, n_layer=12)": GPTConfig(
        vocab_size=50257, block_size=max(context_sizes), n_embd=768, n_head=12, n_layer=12, dropout=0.1
    ),
    "Large (n_embd=1024, n_layer=18)": GPTConfig(
        vocab_size=50257, block_size=max(context_sizes), n_embd=1024, n_head=16, n_layer=18, dropout=0.1
    ),
}

def profile_pass(model, context_size):
    """Profiles a forward and a forward+backward pass for a given model and context."""
    use_amp = any(p.dtype in [torch.bfloat16, torch.float16] for p in model.parameters())
    
    # --- Inference profiling ---
    model.eval()
    inf_total_time = 0
    for i in range(warmup_runs + test_runs):
        input_ids = torch.randint(0, model.config.vocab_size, (1, context_size), device=device)
        if device == 'cuda': torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
            _ = model(input_ids)
        if device == 'cuda': torch.cuda.synchronize()
        end_time = time.perf_counter()
        if i >= warmup_runs: inf_total_time += (end_time - start_time)
    avg_inf_time = (inf_total_time / test_runs) * 1000  # in ms

    # --- Training step profiling ---
    model.train()
    train_total_time = 0
    for i in range(warmup_runs + test_runs):
        input_ids = torch.randint(0, model.config.vocab_size, (1, context_size), device=device)
        if device == 'cuda': torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
            _, loss = model(input_ids, targets=input_ids)
        loss.backward()
        if device == 'cuda': torch.cuda.synchronize()
        end_time = time.perf_counter()
        if i >= warmup_runs: train_total_time += (end_time - start_time)
    avg_train_time = (train_total_time / test_runs) * 1000  # in ms
    
    return avg_inf_time, avg_train_time

# --- Main Analysis Loop ---
results = {}
print(f"--- Performance Analyzer ---")
print(f"Quantization Mode: {QUANTIZATION_MODE}")
if device == 'cuda':
    print(f"Running on: {torch.cuda.get_device_name(device)}")

for name, config in model_configs.items():
    print(f"\nAnalyzing Model: {name}")
    # Create a fresh model instance
    model = GPT(config).to(device)
    
    # Apply quantization if specified
    if QUANTIZATION_MODE == 'bf16_partial':
        model = convert_to_partial_16bit(model, verbose=False)

    # Check if model fits on GPU, skip if it doesn't
    try:
        param_mem_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)
        if device == 'cuda' and param_mem_gb > torch.cuda.get_device_properties(device).total_memory / (1024**3):
             print(f"Skipping model, estimated size ({param_mem_gb:.2f} GB) exceeds GPU memory.")
             continue
    except Exception:
        pass # Continue even if memory check fails

    results[name] = {'inference': [], 'training': []}
    for context in context_sizes:
        print(f"  - Profiling context size: {context}")
        try:
            inf_time, train_time = profile_pass(model, context)
            results[name]['inference'].append(inf_time)
            results[name]['training'].append(train_time)
        except torch.cuda.OutOfMemoryError:
            print(f"    Out of Memory at context size {context}. Stopping analysis for this model.")
            # Fill remaining results with NaN so the plot line stops
            results[name]['inference'].extend([float('nan')] * (len(context_sizes) - len(results[name]['inference'])))
            results[name]['training'].extend([float('nan')] * (len(context_sizes) - len(results[name]['training'])))
            break # Stop testing this model

# --- Plotting Results ---
print("\nGenerating plots...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
plot_title_suffix = f"({QUANTIZATION_MODE} mode)"

# Plot Inference Time
ax1.set_title(f'Inference Time vs. Context Size {plot_title_suffix}')
ax1.set_xlabel('Context Size')
ax1.set_ylabel('Time per Pass (ms)')
for name, data in results.items():
    ax1.plot(context_sizes, data['inference'], marker='o', linestyle='-', label=name)
ax1.grid(True)
ax1.legend()

# Plot Training Step Time
ax2.set_title(f'Training Step vs. Context Size {plot_title_suffix}')
ax2.set_xlabel('Context Size')
ax2.set_ylabel('Time per Step (ms)')
for name, data in results.items():
    ax2.plot(context_sizes, data['training'], marker='o', linestyle='-', label=name)
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.savefig(f'performance_analysis_{QUANTIZATION_MODE}.png')
print(f"Saved performance analysis plots to performance_analysis_{QUANTIZATION_MODE}.png")