import torch
import torch.nn as nn

def convert_to_partial_16bit(model, verbose=True):
    """
    Converts a model to bfloat16 precision for supported layers,
    keeping sensitive layers in float32 for training stability.

    This function is designed for training with Automatic Mixed Precision (AMP).
    It converts major computational layers (Linear, Embedding) to bfloat16
    while keeping layers that require high precision (LayerNorm) in float32.

    Args:
        model (nn.Module): The model to convert.
        verbose (bool): If True, prints which layers are converted.

    Returns:
        nn.Module: The modified model with partial 16-bit precision.
    """
    device = next(model.parameters()).device
    if device.type != 'cuda':
        if verbose:
            print("Warning: 16-bit conversion is only effective on CUDA devices. No changes made.")
        return model

    if not torch.cuda.is_bf16_supported():
        if verbose:
            print("Warning: BFloat16 is not supported on this device. No changes made.")
        return model

    if verbose:
        print("--- Applying Partial BFloat16 Conversion ---")
        print(f"Targeting device: {torch.cuda.get_device_name(device)}")

    for name, module in model.named_modules():
        # --- Layers to Convert to bfloat16 ---
        # These are the main workhorses of a transformer and benefit most from bf16.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.to(torch.bfloat16)
            if verbose:
                print(f"- Converted '{name}' ({type(module).__name__}) to bfloat16")
        
        # --- Layers to Keep in float32 for Stability ---
        # LayerNorm and the final language model head are often kept in float32
        # to maintain precision and training stability.
        elif isinstance(module, nn.LayerNorm) or name == 'lm_head':
            module.to(torch.float32)
            if verbose:
                print(f"- Kept '{name}' ({type(module).__name__}) in float32 for stability")

    if verbose:
        print("\nConversion complete. Model is ready for training with torch.cuda.amp.autocast.")
        
    return model

if __name__ == '__main__':
    # Example Usage:
    # Create a dummy model to demonstrate the conversion.
    # In a real scenario, you would import your GPT model here.
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(100, 64)
            self.linear1 = nn.Linear(64, 128)
            self.ln = nn.LayerNorm(128)
            self.linear2 = nn.Linear(128, 100)
            self.lm_head = self.linear2 # Weight tying

        def forward(self, x):
            return self.lm_head(self.ln(self.linear1(self.embedding(x))))

    # --- Check on CPU (should do nothing) ---
    print("--- Testing on CPU ---")
    cpu_model = SimpleModel()
    convert_to_partial_16bit(cpu_model)
    print(f"CPU Linear Layer dtype: {cpu_model.linear1.weight.dtype}")


    # --- Check on GPU ---
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        print("\n--- Testing on CUDA GPU ---")
        gpu_model = SimpleModel().to('cuda')
        convert_to_partial_16bit(gpu_model)
        
        print("\n--- Verifying Dtypes ---")
        print(f"Embedding Layer dtype : {gpu_model.embedding.weight.dtype}")
        print(f"Linear Layer 1 dtype  : {gpu_model.linear1.weight.dtype}")
        print(f"LayerNorm Layer dtype : {gpu_model.ln.weight.dtype}")
        print(f"LM Head Layer dtype   : {gpu_model.lm_head.weight.dtype}")
    else:
        print("\nSkipping GPU test: CUDA device with bfloat16 support not available.")
