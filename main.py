import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from bitween import Bitween, QuantizedLinear
from bitween.benchmark import generate_benchmark_report
import os
import triton
import triton.language as tl
import time
from bitween.functional import quantize_rtn, dequantize_rtn
import matplotlib.pyplot as plt
import numpy as np

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
    quantizer = Bitween(model, tokenizer=tokenizer, bits=4, group_size=128)

    # # --- 3. Perform quantization and evaluation ---
    quantized_model = quantizer.quantize(
        evaluate_perplexity=True,
        # calculate_parameters_memory=True,
        eval_samples=5,
        # rtn=True,
        trainable=True,
        nsamples=8,
        batch_size=2,
        seqlen=256,
        cache_to_disk=True,
        max_memory_mb=2048,
        ignore_layers=['lm_head', 'embed_tokens']
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

    import copy
    base_model = torch.load(fp32_path, weights_only=False)
    # checkpointed_model = copy.deepcopy(base_model)
    # checkpointed_model = create_memory_efficient_model(
    #     checkpointed_model,
    #     enable_checkpointing=True,
    #     checkpointing_strategy="efficient_block"
    # )

    # Run the benchmark report
    generate_benchmark_report(fp32_path, quantized_path, dummy_input, device)

    # Run the LoRA benchmark report
    from bitween.lora_benchmark import generate_lora_benchmark_report
    dummy_input_dict = {"input_ids": dummy_input}
    generate_lora_benchmark_report(fp32_path, dummy_input_dict, device, text="FP32")
    generate_lora_benchmark_report(quantized_path, dummy_input_dict, device, text="Quantized")


def benchmark_quantized_layers():
    """
    Benchmark QuantizedLinear layers across different sizes and generate plots.
    """
    print("\n--- Benchmarking Quantized Layers ---")
    
    # Define test configurations
    layer_sizes = [128, 256, 512, 1024,
                    # 2048, 4096
                    ]
    batch_sizes = [1, 
                #    16, 32
                   ]
    warmup_iterations = 10
    benchmark_iterations = 1000
    
    # Results storage
    results = {}
    
    for batch_size in batch_sizes:
        results[batch_size] = {
            'sizes': [],
            'pytorch_times': [],
            'quantized_times': [],
            'speedups': [],
            'max_errors': [],
            'mean_errors': []
        }
        
        print(f"\nBenchmarking with batch_size={batch_size}")
        
        for size in layer_sizes:
            print(f"  Testing {size}x{size} layers...")
            
            # Create layers
            float_layer = torch.nn.Linear(size, size, bias=True, dtype=torch.float32).cuda()
            q_layer = QuantizedLinear.from_float(float_layer, bits=8).cuda()
            
            # Create input tensor
            x = torch.randn(batch_size, size, device='cuda', dtype=torch.float32)
            
            # Warm-up run
            for _ in range(warmup_iterations):
                with torch.no_grad():
                    _ = float_layer(x)
                    _ = q_layer(x)
            
            torch.cuda.synchronize()
            
            # Benchmark PyTorch Linear
            start_time = time.time()
            for _ in range(benchmark_iterations):
                with torch.no_grad():
                    _ = float_layer(x)
            torch.cuda.synchronize()
            pytorch_time = (time.time() - start_time) / benchmark_iterations
            
            # Benchmark Quantized Linear
            start_time = time.time()
            for _ in range(benchmark_iterations):
                with torch.no_grad():
                    _ = q_layer(x)
            torch.cuda.synchronize()
            quantized_time = (time.time() - start_time) / benchmark_iterations
            
            # Calculate errors
            with torch.no_grad():
                y_ref = float_layer(x)
                y_quant = q_layer(x)
                max_error = (y_ref - y_quant).abs().max().item()
                mean_error = (y_ref - y_quant).abs().mean().item()
            
            # Store results
            speedup = pytorch_time / quantized_time
            results[batch_size]['sizes'].append(size)
            results[batch_size]['pytorch_times'].append(pytorch_time * 1000)  # Convert to ms
            results[batch_size]['quantized_times'].append(quantized_time * 1000)  # Convert to ms
            results[batch_size]['speedups'].append(speedup)
            results[batch_size]['max_errors'].append(max_error)
            results[batch_size]['mean_errors'].append(mean_error)
            
            print(f"    PyTorch: {pytorch_time*1000:.3f}ms, Quantized: {quantized_time*1000:.3f}ms")
            print(f"    Speedup: {speedup:.2f}x, Max Error: {max_error:.4f}, Mean Error: {mean_error:.4f}")
            
            # Cleanup
            del float_layer, q_layer, x, y_ref, y_quant
            torch.cuda.empty_cache()
    
    # Generate plots
    _generate_benchmark_plots(results)
    
    return results


def _generate_benchmark_plots(results):
    """
    Generate and save benchmark plots.
    """
    print("\nGenerating benchmark plots...")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('QuantizedLinear Performance Benchmarks', fontsize=16)
    
    colors = ['blue', 'red', 'green']
    markers = ['o', 's', '^']
    
    # Plot 1: Execution Times
    for i, (batch_size, data) in enumerate(results.items()):
        ax1.plot(data['sizes'], data['pytorch_times'], 
                color=colors[i], marker=markers[i], linestyle='--', 
                label=f'PyTorch (batch={batch_size})')
        ax1.plot(data['sizes'], data['quantized_times'], 
                color=colors[i], marker=markers[i], linestyle='-', 
                label=f'Quantized (batch={batch_size})')
    
    ax1.set_xlabel('Layer Size')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Execution Time Comparison')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speedup
    for i, (batch_size, data) in enumerate(results.items()):
        ax2.plot(data['sizes'], data['speedups'], 
                color=colors[i], marker=markers[i], 
                label=f'Batch Size {batch_size}')
    
    ax2.set_xlabel('Layer Size')
    ax2.set_ylabel('Speedup (x)')
    ax2.set_title('Quantized vs PyTorch Speedup')
    ax2.set_xscale('log', base=2)
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='No speedup')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Max Error
    for i, (batch_size, data) in enumerate(results.items()):
        ax3.plot(data['sizes'], data['max_errors'], 
                color=colors[i], marker=markers[i], 
                label=f'Batch Size {batch_size}')
    
    ax3.set_xlabel('Layer Size')
    ax3.set_ylabel('Max Absolute Error')
    ax3.set_title('Maximum Quantization Error')
    ax3.set_xscale('log', base=2)
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Mean Error
    for i, (batch_size, data) in enumerate(results.items()):
        ax4.plot(data['sizes'], data['mean_errors'], 
                color=colors[i], marker=markers[i], 
                label=f'Batch Size {batch_size}')
    
    ax4.set_xlabel('Layer Size')
    ax4.set_ylabel('Mean Absolute Error')
    ax4.set_title('Mean Quantization Error')
    ax4.set_xscale('log', base=2)
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('benchmarks', exist_ok=True)
    plot_path = 'benchmarks/quantized_linear_benchmark.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Benchmark plot saved to: {plot_path}")
    
    # Also save as PDF
    pdf_path = 'benchmarks/quantized_linear_benchmark.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Benchmark plot saved to: {pdf_path}")
    
    plt.show()
    
    # Generate summary table
    _generate_summary_table(results)


def _generate_summary_table(results):
    """
    Generate and save a summary table of benchmark results.
    """
    print("\n--- Benchmark Summary ---")
    
    # Create summary table
    summary_lines = []
    summary_lines.append("| Batch Size | Layer Size | PyTorch (ms) | Quantized (ms) | Speedup | Max Error | Mean Error |")
    summary_lines.append("|------------|------------|--------------|----------------|---------|-----------|------------|")
    
    for batch_size, data in results.items():
        for i, size in enumerate(data['sizes']):
            pytorch_time = data['pytorch_times'][i]
            quantized_time = data['quantized_times'][i] 
            speedup = data['speedups'][i]
            max_error = data['max_errors'][i]
            mean_error = data['mean_errors'][i]
            
            line = f"| {batch_size:10d} | {size:10d} | {pytorch_time:12.3f} | {quantized_time:14.3f} | {speedup:7.2f} | {max_error:9.4f} | {mean_error:10.4f} |"
            summary_lines.append(line)
            print(line)
    
    # Save summary to file
    os.makedirs('benchmarks', exist_ok=True)
    summary_path = 'benchmarks/benchmark_summary.md'
    with open(summary_path, 'w') as f:
        f.write("# QuantizedLinear Benchmark Results\n\n")
        f.write("\n".join(summary_lines))
    
    print(f"\nBenchmark summary saved to: {summary_path}")
    
    # Print overall statistics
    print("\n--- Overall Statistics ---")
    all_speedups = []
    all_max_errors = []
    all_mean_errors = []
    
    for data in results.values():
        all_speedups.extend(data['speedups'])
        all_max_errors.extend(data['max_errors'])
        all_mean_errors.extend(data['mean_errors'])
    
    print(f"Average Speedup: {np.mean(all_speedups):.2f}x (±{np.std(all_speedups):.2f})")
    print(f"Best Speedup: {np.max(all_speedups):.2f}x")
    print(f"Worst Speedup: {np.min(all_speedups):.2f}x")
    print(f"Average Max Error: {np.mean(all_max_errors):.4f} (±{np.std(all_max_errors):.4f})")
    print(f"Average Mean Error: {np.mean(all_mean_errors):.4f} (±{np.std(all_mean_errors):.4f})")


if __name__ == "__main__":
    # benchmark_results = benchmark_quantized_layers()
    
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
