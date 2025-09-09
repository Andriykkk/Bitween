#!/usr/bin/env python3
"""Simple quantization test that runs the main function and validates output."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys
import re
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

# Add parent directory to path to import bitween
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bitween import Bitween, QuantizedLinear
from bitween.benchmark import generate_benchmark_report

def run_quantization_test():
    """Run the quantization process and capture output."""
    print("Running quantization test...")
    
    # Capture stdout to parse metrics
    output_buffer = StringIO()
    
    try:
        with redirect_stdout(output_buffer):
            # Copy the main function logic here
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
            
            # Initialize the quantizer
            quantizer = Bitween(model, tokenizer=tokenizer, bits=4, group_size=64)

            # Perform quantization and evaluation
            quantized_model = quantizer.quantize(
                evaluate_perplexity=True,
                eval_samples=5,
                rtn=True,
                nsamples=12,
                batch_size=2,
                seqlen=256,
                cache_to_disk=True,
                max_memory_mb=2048,
                ignore_layers=['lm_head', 'embed_tokens']
            )

            # Save models and run benchmark
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
            from bitween.lora_benchmark import generate_lora_benchmark_report
            dummy_input_dict = {"input_ids": dummy_input}
            generate_lora_benchmark_report(fp32_path, dummy_input_dict, device, text="FP32")
            generate_lora_benchmark_report(quantized_path, dummy_input_dict, device, text="Quantized")
            
    except Exception as e:
        print(f"❌ Quantization failed: {str(e)}")
        return False, ""
    
    output = output_buffer.getvalue()
    return True, output
def run_main_and_parse():
    """Run quantization and parse metrics."""
    success, output = run_quantization_test()
    
    if not success:
        return False
    
    # Parse key metrics
    metrics = {}
    
    # Parse perplexity values
    ppl_orig_match = re.search(r'Original Model PPL:\s+([\d.]+)', output)
    ppl_quant_match = re.search(r'Quantized Model PPL:\s+([\d.]+)', output)
    ppl_diff_match = re.search(r'Difference \(Quant-Orig\):\s+([+-]?[\d.]+)\s+\(([+-]?[\d.]+)%\)', output)
    
    if ppl_orig_match:
        metrics['perplexity_original'] = float(ppl_orig_match.group(1))
    if ppl_quant_match:
        metrics['perplexity_quantized'] = float(ppl_quant_match.group(1))
    if ppl_diff_match:
        metrics['perplexity_difference'] = float(ppl_diff_match.group(1))
        metrics['perplexity_difference_percent'] = float(ppl_diff_match.group(2))
    
    # Parse KL divergence
    kl_div_match = re.search(r'Average KL-Divergence:\s+([\d.]+)', output)
    kl_div_token_match = re.search(r'Average per-token KL-Divergence:\s+([\d.]+)', output)
    
    if kl_div_match:
        metrics['kl_divergence'] = float(kl_div_match.group(1))
    if kl_div_token_match:
        metrics['kl_divergence_per_token'] = float(kl_div_token_match.group(1))
    
    # Parse performance metrics
    memory_reduction_match = re.search(r'Memory Reduction:\s+([\d.]+)%', output)
    throughput_change_match = re.search(r'Throughput Change:\s+([+-]?[\d.]+)%', output)
    
    if memory_reduction_match:
        metrics['memory_reduction_percent'] = float(memory_reduction_match.group(1))
    if throughput_change_match:
        metrics['throughput_change_percent'] = float(throughput_change_match.group(1))
    
    # Parse detailed performance metrics
    fp32_matches = re.findall(r'1\. Benchmarking FP32 Model\.\.\..*?Memory:\s+([\d.]+)\s+MB.*?Throughput:\s+([\d.]+)\s+samples/sec', output, re.DOTALL)
    quant_matches = re.findall(r'2\. Benchmarking Quantized Model\.\.\..*?Memory:\s+([\d.]+)\s+MB.*?Throughput:\s+([\d.]+)\s+samples/sec', output, re.DOTALL)
    
    if fp32_matches:
        metrics['fp32_memory_mb'] = float(fp32_matches[0][0])
        metrics['fp32_throughput'] = float(fp32_matches[0][1])
    
    if quant_matches:
        metrics['quantized_memory_mb'] = float(quant_matches[0][0])
        metrics['quantized_throughput'] = float(quant_matches[0][1])
    
    return validate_and_report(metrics, output)

def validate_and_report(metrics, full_output):
    """Validate metrics and print report."""
    print("\n" + "="*60)
    print("📊 QUANTIZATION TEST RESULTS")
    print("="*60)
    
    # Print parsed metrics
    if 'perplexity_original' in metrics:
        print(f"Original Perplexity:     {metrics['perplexity_original']:.4f}")
    if 'perplexity_quantized' in metrics:
        print(f"Quantized Perplexity:    {metrics['perplexity_quantized']:.4f}")
    if 'perplexity_difference' in metrics:
        print(f"Perplexity Increase:     {metrics['perplexity_difference']:.4f} ({metrics.get('perplexity_difference_percent', 0):.2f}%)")
    
    if 'kl_divergence' in metrics:
        print(f"KL Divergence:           {metrics['kl_divergence']:.6f}")
    if 'kl_divergence_per_token' in metrics:
        print(f"KL Divergence per Token: {metrics['kl_divergence_per_token']:.6f}")
    
    print(f"\n⚡ PERFORMANCE METRICS:")
    if 'memory_reduction_percent' in metrics:
        print(f"Memory Reduction:        {metrics['memory_reduction_percent']:.2f}%")
    if 'throughput_change_percent' in metrics:
        print(f"Throughput Change:       {metrics['throughput_change_percent']:.2f}%")
    
    if 'fp32_memory_mb' in metrics and 'quantized_memory_mb' in metrics:
        print(f"FP32 Memory:            {metrics['fp32_memory_mb']:.2f} MB")
        print(f"Quantized Memory:       {metrics['quantized_memory_mb']:.2f} MB")
    
    if 'fp32_throughput' in metrics and 'quantized_throughput' in metrics:
        print(f"FP32 Throughput:        {metrics['fp32_throughput']:.2f} samples/sec")
        print(f"Quantized Throughput:   {metrics['quantized_throughput']:.2f} samples/sec")
    
    # Simple validation
    errors = []
    warnings = []
    
    # Check perplexity increase (should be reasonable)
    if 'perplexity_difference' in metrics:
        if metrics['perplexity_difference'] > 5.0:
            errors.append(f"Perplexity increase too high: {metrics['perplexity_difference']:.3f}")
        elif metrics['perplexity_difference'] > 2.0:
            warnings.append(f"Perplexity increase is significant: {metrics['perplexity_difference']:.3f}")
    
    # Check KL divergence (should be low)
    if 'kl_divergence' in metrics:
        if metrics['kl_divergence'] > 15.0:
            errors.append(f"KL divergence too high: {metrics['kl_divergence']:.3f}")
        elif metrics['kl_divergence'] > 10.0:
            warnings.append(f"KL divergence is high: {metrics['kl_divergence']:.3f}")
    
    # Performance warnings
    if 'memory_reduction_percent' in metrics and metrics['memory_reduction_percent'] < 30.0:
        warnings.append(f"Low memory reduction: {metrics['memory_reduction_percent']:.2f}%")
    
    if 'throughput_change_percent' in metrics and metrics['throughput_change_percent'] < -70.0:
        warnings.append(f"Significant throughput loss: {metrics['throughput_change_percent']:.2f}%")
    
    print(f"\n🧪 VALIDATION:")
    if errors:
        print("❌ ERRORS:")
        for error in errors:
            print(f"   • {error}")
    
    if warnings:
        print("⚠️  WARNINGS:")
        for warning in warnings:
            print(f"   • {warning}")
    
    if not errors and not warnings:
        print("✅ All metrics look good!")
    
    print("="*60)
    
    # Return success if no errors
    return len(errors) == 0

if __name__ == "__main__":
    success = run_main_and_parse()
    sys.exit(0 if success else 1)