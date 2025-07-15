import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import time
from model import GPT, GPTConfig
from utils import load_model_for_analysis


# --- Configuration (should match main.py) ---
config = GPTConfig(
    vocab_size=50257,
    block_size=1024,  # Max context size the model can handle
    n_embd=384,
    n_head=6,
    n_layer=6,
    dropout=0.1
)

# --- Profiler Settings ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = 'model_final.pt'
# <<< CHANGE THIS TO TEST DIFFERENT MODELS >>>
QUANTIZATION_MODE = 'bf16_partial' # Options: 'none', 'bf16_partial'

# Test configurations: (context_size, tokens_to_generate)
test_cases = [
    (64, 10),
    (128, 10),
    (256, 10),
    (512, 20),
    (1024, 20),
]
warmup_runs = 5
test_runs = 20

# --- Load Model using the utility function ---
model = load_model_for_analysis(
    model_config=config,
    model_path=model_path,
    quantization_type=QUANTIZATION_MODE,
    device=device
)

def profile_pass(mode, context_size, num_tokens):
    """Profiles a forward or forward+backward pass."""
    if mode not in ['inference', 'training']:
        raise ValueError("Mode must be 'inference' or 'training'")

    if mode == 'inference':
        model.eval()
    else:
        model.train()

    total_time = 0
    for i in range(warmup_runs + test_runs):
        # Create dummy data for each run
        input_ids = torch.randint(0, config.vocab_size, (1, context_size), device=device)
        
        # Synchronize CUDA device before starting the timer
        if device == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()

        if mode == 'inference':
            with torch.no_grad():
                for _ in range(num_tokens):
                    _ = model(input_ids) # Forward pass
        else: # training
            for _ in range(num_tokens):
                logits, loss = model(input_ids, targets=input_ids)
                loss.backward() # Backward pass
                # No optimizer step to isolate pass speed

        if device == 'cuda':
            torch.cuda.synchronize()
            
        end_time = time.perf_counter()

        # Only record time for test runs, not warmup
        if i >= warmup_runs:
            total_time += (end_time - start_time)
            
    return total_time / test_runs

# --- Run Profiling ---
print(f"\n--- Performance Profiler ---")
print(f"Device: {torch.cuda.get_device_name(device) if device == 'cuda' else 'CPU'}")
print(f"Warmup Runs: {warmup_runs}, Test Runs: {test_runs}\n")

print(f"{'Context':>10} | {'Gen Tokens':>12} | {'Inference Time/Token (ms)':>28} | {'Training Step Time/Token (ms)':>30}")
print("-" * 90)

for context, gen_tokens in test_cases:
    # Profile Inference
    avg_inference_time = profile_pass('inference', context, gen_tokens)
    inference_time_per_token = (avg_inference_time / gen_tokens) * 1000 # in ms

    # Profile Training
    avg_training_time = profile_pass('training', context, gen_tokens)
    training_time_per_token = (avg_training_time / gen_tokens) * 1000 # in ms

    print(f"{context:>10} | {gen_tokens:>12} | {inference_time_per_token:>28.3f} | {training_time_per_token:>30.3f}")

print("-" * 90)
