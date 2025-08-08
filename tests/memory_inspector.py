import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from model import GPT, GPTConfig
from utils import load_model_for_analysis

# --- Configuration (should match main.py) ---
config = GPTConfig(
    vocab_size=50257,
    block_size=1024,  # Context length for the forward pass test
    n_embd=384,
    n_head=6,
    n_layer=6,
    dropout=0.1
)

# --- Inspector Settings ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = 'model_final.pt'
# <<< CHANGE THIS TO TEST DIFFERENT MODELS >>>
QUANTIZATION_MODE = 'none' # Options: 'none', 'bf16_partial'


def bytes_to_mb(b):
    """Converts bytes to megabytes and formats it."""
    return f"{b / (1024**2):.3f} MB"

# --- Load Model using the utility function ---
model = load_model_for_analysis(
    model_config=config,
    model_path=model_path,
    quantization_type=QUANTIZATION_MODE,
    device=device,
    print_summary=False
)

# 1. --- Measure Parameter Memory ---
# Iterate through each parameter in the model's state_dict and sum up the memory
param_memory = 0
total_params = 0
for param in model.parameters():
    param_memory += param.numel() * param.element_size()
    total_params += param.numel()

bits_per_param = (param_memory * 8) / total_params

print(f"\n--- Static Memory Analysis ---")
print(f"Total Parameters Count : {total_params / 1e6:.2f}M")
print(f"Bits per Parameter     : {bits_per_param:.2f} bits")
print(f"Model Parameters Memory: {bytes_to_mb(param_memory)}")

# --- Initialize Gradients and Optimizer States for Analysis ---
print("\n--- Initializing States for Analysis ---")
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
import bitsandbytes as bnb
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-4)
dummy_input = torch.randint(0, config.vocab_size, (1, config.block_size), device=device)

# Determine if we need to use autocast for mixed precision
use_amp = any(p.dtype in [torch.bfloat16, torch.float16] for p in model.parameters())

print("Performing dummy forward/backward pass...")
# Use autocast to handle mixed precision during the forward pass
with torch.cuda.amp.autocast(enabled=use_amp):
    _, dummy_loss = model(dummy_input, targets=dummy_input)

# This populates gradients and allows optimizer step
dummy_loss.backward()
optimizer.step()
print("Done.")

# 2. --- Measure Gradient Memory ---
gradient_memory = 0
total_grad_elements = 0
grad_null_count = 0
grad_not_required_count = 0
for param in model.parameters():
    if param.grad is not None and param.requires_grad:
        gradient_memory += param.grad.numel() * param.grad.element_size()
        total_grad_elements += param.grad.numel()
    elif param.grad is None and param.requires_grad:
        grad_null_count += 1
    elif param.grad is not None and not param.requires_grad:
        grad_not_required_count += 1

bits_per_grad = (gradient_memory * 8) / total_grad_elements if total_grad_elements > 0 else 0

print(f"Gradients Memory       : {bytes_to_mb(gradient_memory)}")
print(f"Bits per Gradient      : {bits_per_grad:.2f} bits")


# 3. --- Measure Optimizer State Memory ---
optimizer_memory = 0
total_optimizer_elements = 0

total_optimised = 0
total_unoptimised = 0
for state in optimizer.state.values():
    for s in state.values():
        if isinstance(s, torch.Tensor):
            optimizer_memory += s.numel() * s.element_size() 
            total_optimizer_elements += s.numel()
            
bits_per_optim_state = (optimizer_memory * 8) / total_optimizer_elements if total_optimizer_elements > 0 else 0

print(f"Optimizer States Memory  : {bytes_to_mb(optimizer_memory)}")
print(f"Bits per Optimizer State: {bits_per_optim_state:.2f} bits")
print("-" * 35)

total_training_memory = param_memory + gradient_memory + optimizer_memory
print(f"Total Static Memory      : {bytes_to_mb(total_training_memory)}")
print("-" * 35)

# Calculate combined average bits per value for parameters, gradients, and optimizer states
total_elements_all = total_params + total_grad_elements + total_optimizer_elements
total_bits_all = (param_memory + gradient_memory + optimizer_memory) * 8
average_bits_all = total_bits_all / total_elements_all if total_elements_all > 0 else 0

print(f"Average Bits per Value (Parameters + Gradients + Optimizer States): {average_bits_all:.2f} bits")
print("-" * 35)


# 4. --- Measure Activation Memory (Inference) ---
print(f"\n--- Dynamic Memory Analysis (Inference) ---")
print(f"Running forward pass with batch_size=1, sequence_length={config.block_size}...")

model.eval()
activation_memory = 0
if device == 'cuda':
    torch.cuda.reset_peak_memory_stats(device)
    # Ensure memory is clean before measurement
    torch.cuda.empty_cache()
    start_mem = torch.cuda.memory_allocated(device)

# Determine if we need to use autocast for mixed precision
use_amp = any(p.dtype in [torch.bfloat16, torch.float16] for p in model.parameters())

with torch.no_grad():
    # Create a dummy input tensor on the correct device
    input_tensor = torch.randint(0, config.vocab_size, (1, config.block_size), device=device)
    # Use autocast to handle mixed precision during the forward pass
    with torch.cuda.amp.autocast(enabled=use_amp):
        _ = model(input_tensor)

# Measure the peak memory usage during the forward pass
peak_mem = torch.cuda.max_memory_allocated(device)
activation_memory = peak_mem - start_mem

if device == 'cuda':
    print(f"Peak Activation Memory   : {bytes_to_mb(activation_memory)}")
    print(f"(Measured on {torch.cuda.get_device_name(device)})")
else:
    print("Activation memory measurement is only available for CUDA devices.")

print("\nNote: Activation memory is the 'working memory' needed for a forward pass.")
print("-" * 35)
print(f"Total inference memory is (Parameters + Activations): {bytes_to_mb(param_memory + activation_memory + gradient_memory + optimizer_memory)}")
print("-" * 35) # Adjusted to only include parameters and activations for inference

# 5. --- Measure Gradient Memory (Backward Pass) ---
# The gradient memory is already measured above in section 2, but these counts are useful.
print("="*45)
print(f"Null gradient count: {grad_null_count}")
print(f"Gradient not required count: {grad_not_required_count}")