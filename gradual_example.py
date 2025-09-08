"""
Example usage of the gradual quantization system.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the gradual quantization system
from bitween.gradual import GradualQuantizer


def example_gradual_quantization():
    """
    Example of using the gradual quantization system.
    """
    
    # Load model and tokenizer (example with small model)
    model_name = "facebook/opt-125m"  # Small model for testing
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Initialize gradual quantizer
    gradual_quantizer = GradualQuantizer(
        model=model,
        tokenizer=tokenizer,
        target_memory_reduction=0.6,      # Target 60% memory reduction
        max_perplexity_increase=0.05,     # Allow max 5% perplexity increase  
        safety_multiplier=2.0,            # Use 2x budget for working room
        calibration_samples=50,           # Samples for importance analysis
        evaluation_samples=100            # Samples for final evaluation
    )
    
    # Run gradual quantization
    print("Starting gradual quantization...")
    quantized_model = gradual_quantizer.quantize()
    
    # Get detailed report
    report = gradual_quantizer.get_quantization_report()
    memory_analysis = gradual_quantizer.get_memory_analysis()
    
    print("Quantization completed!")
    print(f"Memory reduction: {memory_analysis['reduction_percent']:.1f}%")
    print(f"Perplexity increase: {report['perplexity_increase_percent']:.2f}%")
    
    return quantized_model


def example_step_by_step():
    """
    Example showing step-by-step usage of individual components.
    """
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    
    # Initialize components individually
    from bitween.gradual import ImportanceAnalyzer, PrecisionOptimizer, GradualEvaluator
    
    # 1. Analyze importance
    importance_analyzer = ImportanceAnalyzer(model, tokenizer, calibration_samples=50)
    
    print("Analyzing block importance...")
    block_importance = importance_analyzer.analyze_block_importance()
    
    print("Analyzing layer importance...")  
    layer_importance = importance_analyzer.analyze_layer_importance()
    
    # Convert to thresholds
    global_budget = 0.05
    block_thresholds = importance_analyzer.get_block_thresholds(global_budget)
    
    # 2. Optimize precision
    precision_optimizer = PrecisionOptimizer(model, tokenizer)
    
    print("Optimizing block precisions...")
    for block_name, threshold in block_thresholds.items():
        print(f"  Optimizing {block_name} with threshold {threshold:.4f}")
        result = precision_optimizer.progressive_block_quantization(block_name, threshold)
        print(f"    Result: {result[-1].result.value} at {result[-1].target_bits} bits")
    
    # 3. Evaluate results
    evaluator = GradualEvaluator(model, tokenizer)
    
    print("Evaluating quantized model...")
    final_metrics = evaluator.comprehensive_evaluation()
    
    print(f"Final perplexity: {final_metrics.perplexity:.4f}")
    print(f"Memory usage: {final_metrics.memory_usage_mb:.1f} MB")
    
    return model


if __name__ == "__main__":
    # Run simple example
    quantized_model = example_gradual_quantization()
    
    # Or run step-by-step example
    # quantized_model = example_step_by_step()