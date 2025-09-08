# Gradual Quantization System

A sophisticated adaptive quantization system that progressively reduces model precision based on layer importance analysis and perplexity budget management.

## Overview

The gradual quantization system addresses the key challenge in model quantization: **how to minimize memory usage while preserving model quality**. It does this through:

1. **Multi-method importance analysis** to identify which layers can tolerate aggressive quantization
2. **Progressive precision reduction** (8-bit → 4-bit → 2-bit) with automatic rollback
3. **Dynamic group size optimization** for each precision level  
4. **Budget-aware quantization** with safety margins and adaptive thresholds

## Architecture

### Core Components

- **`GradualQuantizer`**: Main orchestrator that coordinates the entire process
- **`ImportanceAnalyzer`**: Discovers layer/block importance through multiple methods
- **`PrecisionOptimizer`**: Handles quantization with precision and group size optimization
- **`GradualEvaluator`**: Specialized evaluation utilities for the gradual process

### Key Features

- **Clean separation** from existing `Bitween` quantizer
- **Multiple importance discovery methods**: noise injection, RTN sensitivity, block disable, layer upgrade
- **Automatic rollback** when quantization exceeds quality thresholds
- **Memory-efficient** processing with minimal peak memory usage
- **Comprehensive evaluation** with perplexity, KL-divergence, and memory analysis

## Usage

### Simple Usage
```python
from bitween.gradual import GradualQuantizer

# Initialize quantizer
quantizer = GradualQuantizer(
    model=model,
    tokenizer=tokenizer, 
    target_memory_reduction=0.6,     # 60% memory reduction
    max_perplexity_increase=0.05     # Max 5% perplexity increase
)

# Run gradual quantization
quantized_model = quantizer.quantize()

# Get results
report = quantizer.get_quantization_report()
memory_analysis = quantizer.get_memory_analysis()
```

### Advanced Usage
```python 
from bitween.gradual import ImportanceAnalyzer, PrecisionOptimizer, GradualEvaluator

# Step-by-step process
importance_analyzer = ImportanceAnalyzer(model, tokenizer)
block_importance = importance_analyzer.analyze_block_importance()
layer_importance = importance_analyzer.analyze_layer_importance()

# Get adaptive thresholds  
thresholds = importance_analyzer.get_block_thresholds(global_budget=0.05)

# Optimize each block
precision_optimizer = PrecisionOptimizer(model, tokenizer)
for block_name, threshold in thresholds.items():
    result = precision_optimizer.progressive_block_quantization(block_name, threshold)
    print(f"{block_name}: {result[-1].target_bits} bits")

# Comprehensive evaluation
evaluator = GradualEvaluator(model, tokenizer)
metrics = evaluator.comprehensive_evaluation()
```

## Algorithm Overview

### Phase 1: Importance Discovery
- **Noise injection testing**: Add random noise and measure impact
- **RTN sensitivity testing**: Test 8/4/2-bit RTN quantization impact
- **Block disable testing**: Measure impact of completely disabling blocks
- **Layer upgrade testing**: Quantize to 4-bit baseline, then test upgrading layers

### Phase 2: Progressive Quantization  
- Process blocks in order of increasing importance (least important first)
- For each block, try 8-bit → 4-bit → 2-bit quantization
- For each precision, optimize group size (128 → 64 → 32 → 16)
- Use trainable quantization as fallback when RTN fails
- Rollback if perplexity budget exceeded

### Phase 3: Global Optimization
- Apply final precision configuration to entire model
- End-to-end validation against user quality thresholds
- Selective precision upgrades if global budget violated

## Configuration

### Key Parameters

- **`target_memory_reduction`**: Desired memory reduction (0.0-1.0)
- **`max_perplexity_increase`**: Maximum allowed perplexity increase
- **`safety_multiplier`**: Multiply budget for working room (default: 2.0)
- **`calibration_samples`**: Samples for importance analysis (default: 100)
- **`evaluation_samples`**: Samples for final evaluation (default: 200)

### Importance Method Weights

- **RTN sensitivity**: 40% (most reliable indicator)
- **Layer upgrade**: 25% (direct quantization relevance)
- **Block disable**: 20% (criticality assessment)
- **Noise injection**: 15% (robustness measure)

## Implementation Status

### ✅ Completed
- Architecture design and class structure
- Method signatures and documentation
- Integration with existing system
- Example usage code

### 🚧 To Implement
- Importance analysis methods
- Precision optimization algorithms
- Evaluation utilities  
- Group size optimization
- State management and rollback
- Memory profiling and analysis

## Directory Structure

```
bitween/gradual/
├── __init__.py                 # Module exports
├── gradual_quantizer.py        # Main orchestrator
├── importance_analyzer.py      # Importance discovery methods
├── precision_optimizer.py      # Quantization optimization  
├── evaluator.py               # Specialized evaluation
├── utils.py                   # Utility functions
└── README.md                  # This file
```

## Design Philosophy

1. **Memory-first optimization**: Prioritize memory reduction while maintaining quality
2. **Conservative with safety margins**: Use 2x perplexity budget for working room
3. **Multiple validation methods**: Cross-validate importance through different approaches
4. **Graceful degradation**: Always have rollback options when optimization fails
5. **Modular design**: Each component can be used independently for flexibility

## Future Enhancements

- Support for non-transformer architectures
- Integration with pruning and other compression techniques
- Advanced layer cooperation analysis
- Automated hyperparameter tuning
- Support for very low precision (1-bit, ternary)
- Hardware-specific optimizations