import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from bitween import Bitween

def main():
    """
    Main function to demonstrate the use of the MyQuantizer library.
    """
    # --- 1. Load a pre-trained model from Hugging Face ---
    # We use a small model for this example.
    model_name = "sbintuitions/tiny-lm-chat"
    print(f"Loading model: {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("\nOriginal model structure:")
    print(model)

    # --- 2. Initialize the quantizer ---
    # We'll quantize to 4 bits with per-channel quantization (group_size=-1)
    quantizer = Bitween(model, bits=8, group_size=-1)

    # --- 3. Perform quantization ---
    quantized_model = quantizer.quantize()

    # --- 4. Inspect the quantized model ---
    print("\nQuantized model structure:")
    print(quantized_model)
    
    print("\nVerification:")
    print("Notice how 'nn.Linear' layers have been replaced with 'QuantizedLinear'.")


if __name__ == "__main__":
    main()