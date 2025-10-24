# Bitween - High-Performance Neural Network Quantization Library

**Bitween** is a PyTorch quantization library that reduces model size by 2-8x while maintaining accuracy through optimized CUDA kernels with tensor core acceleration.

## Features

- **Multiple quantization methods**: RTN (Round-to-Nearest), Trainable, and Gradual quantization
- **Flexible bit-widths**: 2-bit, 4-bit, and 8-bit quantization support
- **Per-group quantization**: Configurable group sizes (32, 64, 128, ...) for accuracy/compression trade-off
- **Memory efficient**: Reduce model size by 75-87.5% with minimal accuracy loss

> **Note:** Quantization primarily reduces model size, not inference speed. Dequantization and matrix multiplication overhead can negate speed benefits, as cuBLAS (closed-source) is highly optimized for FP16 operations on modern GPUs. Quantization is most useful for memory savings and deploying larger models in limited VRAM.

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd quantisation

# Install dependencies
pip install -r requirements.txt

# The CUDA kernels will auto-compile on first use
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)
- Triton (for fallback kernels)

## Quick Start

### Basic RTN Quantization (Fastest)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from bitween import Bitween

# Load your model
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m").cuda()
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

# Initialize quantizer
quantizer = Bitween(
    model=model,
    tokenizer=tokenizer,
    bits=4,              # 4-bit quantization (75% size reduction)
    group_size=128       # Group size for quantization
)

# Quantize the model
quantized_model = quantizer.quantize(
    rtn=True,                    # Use fast RTN quantization
    evaluate_perplexity=True,    # Evaluate quality
    eval_samples=10,             # Number of evaluation samples
    ignore_layers=['lm_head', 'embed_tokens']  # Skip embedding layers
)

# Save quantized model
import torch
torch.save(quantized_model, "model_quantized.pth")
```

## Quantization Methods

### 1. RTN (Round-to-Nearest) - **Recommended for Speed**

**What it does:**
- Fastest quantization method (seconds for small models)
- Simply rounds weights to nearest quantized value
- No calibration data needed
- Best for: Quick prototyping, inference optimization

**When to use:**
- You need fast quantization
- Slight accuracy loss is acceptable
- No calibration data available

**Example:**
```python
quantized_model = quantizer.quantize(
    rtn=True,                   # Enable RTN mode
    evaluate_perplexity=True,
    eval_samples=10
)
```

**Results (typical):**
- **Speed**: ~5-10 seconds for 125M parameter model
- **Accuracy**: 1-3% perplexity increase
- **Size**: 4-bit = 75% reduction, 8-bit = 50% reduction

---

### 2. Trainable Quantization - **Recommended for Quality**

**What it does:**
- Optimizes quantization parameters (scale, zero-point) using calibration data
- Minimizes reconstruction error through gradient descent
- Uses block-wise training for memory efficiency
- Best for: Maximum accuracy retention

**When to use:**
- You have calibration data (unlabeled text)
- Best possible accuracy is critical
- Can afford ~5-10 minutes training time

**Example:**
```python
quantized_model = quantizer.quantize(
    rtn=False,
    trainable=True,              # Enable trainable mode
    calib_dataset="pile-10k",    # Calibration dataset
    nsamples=128,                # Number of calibration samples
    batch_size=4,                # Training batch size
    evaluate_perplexity=True,
    cache_to_disk=True,          # Save activations to disk (memory efficient)
    max_memory_mb=2048           # Max memory for caching
)
```

**Hyperparameters:**
- `iters`: Number of training epochs per block (default: 1)
- `lr`: Learning rate for optimization (default: 0.005)
- `enable_minmax_tuning`: Optimize scale bounds (default: True)
- `nsamples`: Number of calibration samples (128-512 recommended)
- `batch_size`: Training batch size (2-8 depending on GPU memory)

**Results (typical):**
- **Speed**: ~5-10 minutes for 125M parameter model
- **Accuracy**: 0.5-2% perplexity increase (better than RTN)
- **Size**: Same as RTN (4-bit = 75%, 8-bit = 50%)

---

### 3. Gradual Quantization - **Recommended for Extreme Compression**

**What it does:**
- Automatically determines optimal bit-width per layer
- Progressively reduces precision while monitoring quality
- Allocates budget based on layer importance
- Starts at 8-bit, tries 4-bit, then 2-bit for each layer
- Best for: Maximum compression with quality constraints

**When to use:**
- You want maximum compression
- Different layers have different sensitivity
- You have perplexity/KL-divergence constraints

**Example:**
```python
from bitween.gradual import GradualQuantizer

quantizer = GradualQuantizer(
    model=model,
    tokenizer=tokenizer,
    max_perplexity_increase=5,     # Max 5% perplexity increase
    max_per_token_kl_divergence=0.01,  # KL-div limit per token
    nsamples=128,                  # Calibration samples
    evaluation_samples=5,          # Evaluation samples
    min_group_size=32,             # Min group size
    max_group_size=128,            # Max group size
    ignore_layers=['lm_head', 'embed_tokens']
)

# Automatically determine best quantization per layer
quantized_model = quantizer.quantize()
```

**How it works:**
1. **Baseline**: Measures original model perplexity/KL-divergence
2. **Importance Analysis**: Ranks layers by sensitivity
3. **Budget Allocation**: Distributes quality budget across layers
4. **Progressive Quantization**: For each layer (in order of importance):
   - Try 8-bit → Check if within budget
   - If sucessfull, try 4-bit → Check if within budget
   - If sucessfull, try 2-bit → Check if within budge
5. **Optimization**: Fine-tune quantization parameters
6. **Validation**: Ensure final model meets constraints

**Results (typical):**
- **Speed**: ~15-30 minutes for 125M parameter model
- **Accuracy**: Stays within specified constraints (e.g., 5% PPL increase)
- **Size**: 80-90% reduction (mix of 2/4/8-bit layers)
- **Smart allocation**: Critical layers get more bits, simple layers get fewer

---

## Configuration Guide

### Bit-Width Selection

| Bits | Size Reduction | Accuracy | Use Case |
|------|---------------|----------|----------|
| **8-bit** | 50% | Excellent | Production models, minimal loss |
| **4-bit** | 75% | Good | Balanced compression/quality |
| **2-bit** | 87.5% | Fair | Extreme compression, less critical layers |

**Rule of thumb**: Larger group sizes = smaller model but slightly lower accuracy

### Memory Management

For trainable quantization with limited GPU memory:

```python
quantized_model = quantizer.quantize(
    trainable=True,
    cache_to_disk=True,      # Save activations to disk instead of RAM
    max_memory_mb=1024,      # Limit cache size to 1GB
    batch_size=2             # Smaller batches use less memory
)
```

---

## Advanced Usage

### Custom Layer Filtering

```python
# Quantize only specific layer types
quantizer = Bitween(model, tokenizer, bits=4, group_size=128)

# Skip embedding and output layers (common practice)
quantized_model = quantizer.quantize(
    ignore_layers=['lm_head', 'embed_tokens', 'wte', 'wpe']
)
```

### Performance Benchmarking

```python
from bitween.benchmark import generate_benchmark_report

# Compare FP32 vs Quantized
dummy_input = torch.randint(0, model.config.vocab_size, (1, 128), device='cuda')
generate_benchmark_report(
    "model_fp32.pth",
    "model_quantized.pth",
    dummy_input,
    device='cuda'
)
```

### Fine-tuning Parameters (Trainable Mode)

```python
quantizer = Bitween(
    model, tokenizer,
    bits=4,
    group_size=128,
    lr=0.01,                    # Higher learning rate
    enable_minmax_tuning=True   # Optimize scale bounds
)

quantized_model = quantizer.quantize(
    trainable=True,
    nsamples=256,              # More calibration samples
    batch_size=8               # Larger batches (if GPU allows)
)
```

---

## Typical Workflows

### Workflow 1: Quick Quantization for Deployment

```python
# 1. Load model
model = AutoModelForCausalLM.from_pretrained("your-model").cuda()
tokenizer = AutoTokenizer.from_pretrained("your-model")

# 2. Quick RTN quantization
quantizer = Bitween(model, tokenizer, bits=4, group_size=128)
quantized_model = quantizer.quantize(rtn=True)

# 3. Save and deploy
torch.save(quantized_model, "model_4bit.pth")
```

**Time**: ~10 seconds | **Size**: 75% smaller | **Accuracy**: ~2% PPL increase

---

### Workflow 2: High-Quality Quantization

```python
# 1. Load model
model = AutoModelForCausalLM.from_pretrained("your-model").cuda()
tokenizer = AutoTokenizer.from_pretrained("your-model")

# 2. Trainable quantization with calibration
quantizer = Bitween(model, tokenizer, bits=4, group_size=128)
quantized_model = quantizer.quantize(
    trainable=True,
    calib_dataset="pile-10k",
    nsamples=256,
    batch_size=4,
    evaluate_perplexity=True
)

# 3. Save
torch.save(quantized_model, "model_4bit_trained.pth")
```

**Time**: ~10 minutes | **Size**: 75% smaller | **Accuracy**: ~1% PPL increase

---

### Workflow 3: Maximum Compression

```python
from bitween.gradual import GradualQuantizer

# 1. Load model
model = AutoModelForCausalLM.from_pretrained("your-model").cuda()
tokenizer = AutoTokenizer.from_pretrained("your-model")

# 2. Gradual quantization
quantizer = GradualQuantizer(
    model, tokenizer,
    max_perplexity_increase=5,
    max_per_token_kl_divergence=0.01
)
quantized_model = quantizer.quantize()

# 3. Save
torch.save(quantized_model, "model_gradual.pth")
```

**Time**: ~30 minutes | **Size**: 80-90% smaller | **Accuracy**: Within constraints

---

## Troubleshooting

### CUDA Kernel Compilation Errors

If you get CUDA compilation errors:

```python
# The library will automatically fallback to Triton kernels
# Check if CUDA kernel loaded:
from bitween.kernels.cuda_loader import quantized_matmul_cuda
# If this fails, Triton will be used instead
```

### Out of Memory (OOM)

For trainable quantization on limited GPU:

```python
quantizer.quantize(
    trainable=True,
    cache_to_disk=True,      # Use disk instead of RAM
    max_memory_mb=512,       # Reduce cache size
    batch_size=1,            # Smallest batch size
    nsamples=64              # Fewer calibration samples
)
```

### Poor Accuracy After Quantization

1. **Try trainable mode instead of RTN**
2. **Increase group size**: 64 → 128
3. **Use 8-bit instead of 4-bit**
4. **Increase calibration samples**: nsamples=256
5. **Use gradual quantization** for automatic bit allocation

---

## API Reference

### Bitween Class

```python
Bitween(
    model,                    # PyTorch model to quantize
    tokenizer=None,           # HuggingFace tokenizer
    bits=8,                   # Bit-width (2, 4, or 8)
    group_size=32,            # Quantization group size
    iters=1,                  # Training epochs per block
    lr=0.005,                 # Learning rate
    enable_minmax_tuning=True,# Optimize scale bounds
    seqlen=2048,              # Sequence length
    cache_to_disk=False,      # Disk caching
    max_memory_mb=512,        # Max cache memory
    ignore_layers=None,       # Layers to skip
    batch_size=32             # Training batch size
)
```

### GradualQuantizer Class

```python
GradualQuantizer(
    model,                    # PyTorch model
    tokenizer,                # HuggingFace tokenizer
    max_perplexity_increase=5,# Max PPL increase (%)
    max_per_token_kl_divergence=0.01,  # Max KL-div
    nsamples=128,             # Calibration samples
    evaluation_samples=5,     # Evaluation samples
    ignore_layers=None,       # Layers to skip
    min_group_size=32,        # Min group size
    max_group_size=128,       # Max group size
    training_batch_size=4     # Batch size
)
```

---

## Citation

If you use this library in your research, please cite:

```bibtex
@software{bitween2024,
  title={Bitween: High-Performance Neural Network Quantization},
  author={Andriykkk},
  year={2024},
  url={https://github.com/Andriykkk/Bitween}
}
```

## License

[Your License Here]

## Acknowledgments

- Built with PyTorch and Triton
- CUDA kernel optimization inspired by FlashAttention and GPTQ
- Tensor core programming based on NVIDIA CUTLASS