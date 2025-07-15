import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import sys
from model import GPT, GPTConfig

def get_unique_modules(model):
    """
    Recursively finds all unique module classes in a given model.
    """
    unique_modules = set()
    for module in model.modules():
        # We don't need to check containers, only functional layers
        if len(list(module.children())) == 0 and not isinstance(module, (nn.ModuleDict, nn.ModuleList, nn.Sequential)):
             unique_modules.add(type(module))
    return list(unique_modules)

def check_support(layer_class, dtype, device):
    """
    Checks if a given layer class supports a specific dtype on a device.
    Returns: (is_supported, message)
    """
    try:
        # --- Create a representative instance of the layer ---
        if layer_class == nn.Linear:
            layer = layer_class(16, 32)
            dummy_input = torch.randn(4, 16, device=device)
        elif layer_class == nn.Embedding:
            layer = layer_class(100, 32)
            dummy_input = torch.randint(0, 100, (4, 10), device=device)
        elif layer_class == nn.LayerNorm:
            layer = layer_class(16)
            dummy_input = torch.randn(4, 10, 16, device=device)
        elif layer_class == nn.Dropout:
            # Dropout is a special case; it's usually fine but let's test
            layer = layer_class(p=0.1)
            dummy_input = torch.randn(4, 16, device=device)
        else:
            # Fallback for any other layer types found
            # This might need adjustment if a layer has a very specific input
            try:
                layer = layer_class()
                dummy_input = torch.randn(4, 16, device=device)
            except Exception:
                 return (False, "❓ Needs custom setup")

        layer.to(device).to(dtype)
        dummy_input = dummy_input.to(dtype)

        # --- Run forward pass ---
        with torch.no_grad():
            _ = layer(dummy_input)
            
        return (True, "✅ Supported")

    except RuntimeError as e:
        error_msg = str(e).lower()
        if 'not implemented' in error_msg or 'is not supported' in error_msg or 'expected' in error_msg:
            return (False, f"❌ Not Supported")
        else:
            return (False, f"❓ Error: {str(e)[:60]}...")
    except Exception as e:
        return (False, f"❓ Error: {str(e)[:60]}...")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA device not found. This check is only meaningful on a GPU.")
        sys.exit()

    device = 'cuda'
    gpu_name = torch.cuda.get_device_name(device)
    print(f"--- Checking 16-bit Support for Modules in GPT Model on: {gpu_name} ---\n")

    # --- Instantiate the user's model to inspect its layers ---
    config = GPTConfig(vocab_size=4, block_size=4, n_embd=4, n_head=2, n_layer=1, dropout=0.1)
    model = GPT(config)
    
    modules_to_check = get_unique_modules(model)

    dtypes_to_check = {
        "Float16": torch.float16,
        "BFloat16": torch.bfloat16,
    }

    print(f"{'Module Type':<25} | {'Float16':<20} | {'BFloat16':<20}")
    print("-" * 70)

    for module_cls in modules_to_check:
        results = {}
        for name, dtype in dtypes_to_check.items():
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                results[name] = "N/A (Hardware)"
                continue
            
            supported, msg = check_support(module_cls, dtype, device)
            results[name] = msg

        print(f"{module_cls.__name__:<25} | {results['Float16']:<20} | {results['BFloat16']:<20}")

    print("-" * 70)
    print("\nThis script inspects your GPT model, finds all unique layer types,")
    print("and tests if they support 16-bit operations on your hardware.")
