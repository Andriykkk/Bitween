"""
Compare outputs between original and quantized models.

This script loads both models from temp_models/ and runs inference
to compare their outputs on test prompts.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os


def load_model(model_path, device='cuda'):
    """Load a model from state dict."""
    print(f"Loading model from {model_path}...")

    # Load the model directly (it was saved as a full model, not just state dict)
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    return model


def generate_text(model, tokenizer, prompt, max_length=50, device='cuda'):
    """Generate text from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=False,  # Use greedy decoding for reproducibility
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def compare_models():
    """Compare outputs from original and quantized models."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Paths to saved models
    original_path = "temp_models/opt125m_fp32.pth"
    quantized_path = "temp_models/opt125m_quantized.pth"

    # Check if files exist
    if not os.path.exists(original_path):
        print(f"Error: {original_path} not found!")
        return
    if not os.path.exists(quantized_path):
        print(f"Error: {quantized_path} not found!")
        return

    # Load tokenizer (same for both models)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

    # Test prompts
    test_prompts = [
        "The capital of France is",
        "Artificial intelligence is",
        "Once upon a time",
        "The meaning of life is",
        "In the year 2050, technology will"
    ]

    print("\n" + "="*80)
    print("COMPARING MODEL OUTPUTS")
    print("="*80 + "\n")

    # Load and test original model
    print("--- ORIGINAL MODEL (FP32) ---\n")
    original_model = load_model(original_path, device)

    original_outputs = []
    for prompt in test_prompts:
        output = generate_text(original_model, tokenizer, prompt, max_length=50, device=device)
        original_outputs.append(output)
        print(f"Prompt: {prompt}")
        print(f"Output: {output}\n")

    # Free memory
    del original_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\n" + "-"*80 + "\n")

    # Load and test quantized model
    print("--- QUANTIZED MODEL ---\n")
    quantized_model = load_model(quantized_path, device)

    quantized_outputs = []
    for prompt in test_prompts:
        output = generate_text(quantized_model, tokenizer, prompt, max_length=50, device=device)
        quantized_outputs.append(output)
        print(f"Prompt: {prompt}")
        print(f"Output: {output}\n")

    # Free memory
    del quantized_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\n" + "="*80)
    print("SIDE-BY-SIDE COMPARISON")
    print("="*80 + "\n")

    for i, prompt in enumerate(test_prompts):
        print(f"Prompt: {prompt}\n")
        print(f"Original:  {original_outputs[i]}")
        print(f"Quantized: {quantized_outputs[i]}")

        # Check if outputs match
        if original_outputs[i] == quantized_outputs[i]:
            print("✓ IDENTICAL")
        else:
            print("✗ DIFFERENT")

        print("\n" + "-"*80 + "\n")


if __name__ == "__main__":
    compare_models()
