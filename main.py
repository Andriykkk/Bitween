import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from bitween import Bitween
from bitween.benchmark import generate_benchmark_report
import os

def main():
    """
    Main function to demonstrate the full, integrated pipeline:
    1. Load a pre-trained model and tokenizer.
    2. Initialize the Bitween quantizer.
    3. Call the quantize method with evaluation enabled.
    4. Save models and run benchmark.
    """
    # --- 1. Load a pre-trained model from Hugging Face ---
    model_name = "facebook/opt-125m"
    print(f"Loading model: {model_name}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version PyTorch built with:", torch.version.cuda)
    print("GPU devices:", torch.cuda.device_count())

    model_dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=model_dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # --- 2. Initialize the quantizer ---
    quantizer = Bitween(model, tokenizer=tokenizer, bits=8, group_size=64)

    # --- 3. Perform quantization and evaluation ---
    quantized_model = quantizer.quantize(
        evaluate_perplexity=True,
        num_samples=5
    )

    # --- 4. Save models and run benchmark ---
    print("\n--- Preparing for Benchmark ---")
    save_dir = "temp_models"
    os.makedirs(save_dir, exist_ok=True)
    fp32_path = os.path.join(save_dir, "opt125m_fp32.pth")
    quantized_path = os.path.join(save_dir, "opt125m_quantized.pth")

    print(f"Saving FP32 model to {fp32_path}...")
    torch.save(model, fp32_path)
    
    print(f"Saving quantized model to {quantized_path}...")
    torch.save(quantized_model, quantized_path)

    # Create a dummy input for benchmarking
    dummy_input = torch.randint(0, model.config.vocab_size, (1, 128), device=device)

    # Run the benchmark report
    generate_benchmark_report(fp32_path, quantized_path, dummy_input, device)

    # Run the LoRA benchmark report
    # from bitween.lora_benchmark import generate_lora_benchmark_report
    # dummy_input_dict = {"input_ids": dummy_input}
    # generate_lora_benchmark_report(fp32_path, dummy_input_dict, device, text="FP32")
    # generate_lora_benchmark_report(quantized_path, dummy_input_dict, device, text="Quantized")


if __name__ == "__main__":
    main()