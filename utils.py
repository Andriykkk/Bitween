import torch
import os
from model import GPT, GPTConfig
from quantisation.q16bit_partial import convert_to_partial_16bit

def load_model_for_analysis(model_config, model_path, quantization_type='none', device='cpu'):
    """
    Loads a GPT model for analysis, with optional quantization.

    Args:
        model_config (GPTConfig): The configuration object for the GPT model.
        model_path (str): Path to the saved model checkpoint (.pt file).
        quantization_type (str): The type of quantization to apply. 
                                 Options: 'none', 'bf16_partial'.
        device (str): The device to load the model onto ('cpu' or 'cuda').

    Returns:
        torch.nn.Module: The loaded and configured model.
    """
    print(f"--- Loading Model ---")
    print(f"Quantization: '{quantization_type}', Device: '{device}'")

    model = GPT(model_config).to(device)
    
    if os.path.exists(model_path):
        print(f"Loading checkpoint from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: Model checkpoint not found at '{model_path}'. Using random weights.")

    if quantization_type == 'bf16_partial':
        model = convert_to_partial_16bit(model, verbose=True)
    elif quantization_type != 'none':
        print(f"Warning: Unknown quantization type '{quantization_type}'. No quantization applied.")
    else:
        print("Model loaded with default (Float32) precision.")

    print("-" * 25)
    return model

if __name__ == '__main__':
    # Example of how to use the function
    config = GPTConfig(vocab_size=4, block_size=4, n_embd=4, n_head=2, n_layer=1, dropout=0.1)
    
    # Create a dummy checkpoint for demonstration
    dummy_model = GPT(config)
    dummy_path = "dummy_model.pt"
    torch.save(dummy_model.state_dict(), dummy_path)

    print("\n1. Loading model with no quantization:")
    model_fp32 = load_model_for_analysis(config, dummy_path, 'none')
    print(f"Linear layer dtype: {model_fp32.transformer.h[0].attn.c_attn.weight.dtype}")

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        print("\n2. Loading model with partial bfloat16 quantization:")
        model_bf16 = load_model_for_analysis(config, dummy_path, 'bf16_partial', device='cuda')
        print(f"Linear layer dtype: {model_bf16.transformer.h[0].attn.c_attn.weight.dtype}")
        print(f"LayerNorm dtype: {model_bf16.transformer.h[0].ln_1.weight.dtype}")

    # Clean up the dummy file
    os.remove(dummy_path)
