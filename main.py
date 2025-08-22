import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from bitween import Bitween, QuantizedLinear
from bitween.benchmark import generate_benchmark_report
import os
import triton
import triton.language as tl
import time
from bitween.functional import quantize_rtn, dequantize_rtn

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
    quantizer = Bitween(model, tokenizer=tokenizer, bits=8, group_size=128)

    # --- 3. Perform quantization and evaluation ---
    quantized_model = quantizer.quantize(
        evaluate_perplexity=False,
        num_samples=5,
        rtn=True
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
    # generate_benchmark_report(fp32_path, quantized_path, dummy_input, device)

    # Run the LoRA benchmark report
    from bitween.lora_benchmark import generate_lora_benchmark_report
    dummy_input_dict = {"input_ids": dummy_input}
    generate_lora_benchmark_report(fp32_path, dummy_input_dict, device, text="FP32")
    generate_lora_benchmark_report(quantized_path, dummy_input_dict, device, text="Quantized")


if __name__ == "__main__":
    # Define layer dimensions and batch size/sequence length
    in_features, out_features = 1280, 1280
    batch_size = 32
    
    # Instantiate a float linear layer and a quantized layer
    float_layer = torch.nn.Linear(in_features, out_features, bias=True, dtype=torch.float32).cuda()
    q_layer = QuantizedLinear.from_float(float_layer, bits=8).cuda()
    
    # Create a random input tensor
    x = torch.randn(batch_size, in_features, device='cuda', dtype=torch.float32)

    # Warm-up run to prevent CUDA overhead from affecting measurements
    for _ in range(10):
        y_ref = float_layer(x)
        y_quant = q_layer(x)
        
    # --- Performance comparison ---
    # PyTorch timing
    start_time = time.time()
    for _ in range(1000):
        y_ref = float_layer(x)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start_time) / 100
    
    # Quantized kernel timing
    start_time = time.time()
    for _ in range(1000):
        y_quant = q_layer(x)
    torch.cuda.synchronize()
    quant_time = (time.time() - start_time) / 100
    
    # Sanity check
    y_ref = float_layer(x)
    y_quant = q_layer(x)

    # Print results
    print(f"PyTorch Linear Time: {pytorch_time*1000:.3f} ms")
    print(f"Quantized Kernel Time: {quant_time*1000:.3f} ms")

    print(f"Max Error: {(y_ref - y_quant).abs().max():.4f}")
    print(f"Mean Error: {(y_ref - y_quant).abs().mean():.4f}")
    print(f"Speedup: {pytorch_time / quant_time:.2f}x")
    main()

    # def get_quantized_model_size(model):
    #     total_bytes = 0
    #     layer_info = []

    #     for name, module in model.named_modules():
    #         if isinstance(module, QuantizedLinear):
    #             # qweight
    #             qweight_bytes = module.qweight.numel() * module.qweight.element_size()
    #             # scale
    #             scale_bytes = module.scale.numel() * module.scale.element_size()
    #             # zero_point
    #             zp_bytes = module.zero_point.numel() * module.zero_point.element_size()
    #             # bias
    #             bias_bytes = module.bias.numel() * module.bias.element_size() if module.bias is not None else 0

    #             total = qweight_bytes + scale_bytes + zp_bytes + bias_bytes
    #             total_bytes += total

    #             layer_info.append({
    #                 "layer": name,
    #                 "qweight_MB": qweight_bytes / 1024**2,
    #                 "scale_MB": scale_bytes / 1024**2,
    #                 "zero_point_MB": zp_bytes / 1024**2,
    #                 "bias_MB": bias_bytes / 1024**2,
    #                 "total_MB": total / 1024**2
    #             })
    #         else:
    #             total = 0
    #             for param in module.parameters():
    #                 total += param.numel() * param.element_size()
    #             total_bytes += total

    #             layer_info.append({
    #                 "layer": name,
    #                 "total_MB": total / 1024**2
    #             })

    #     return layer_info, total_bytes / 1024**2

    # # Example usage:
    # layer_breakdown, total_MB = get_quantized_model_size(quantized_model)
    # for info in layer_breakdown:
    #     print(info)
    # print(f"Total quantized model size: {total_MB:.2f} MB")
