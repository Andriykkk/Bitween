import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import QuantizedLinear
from .functional import quantize_rtn, dequantize_rtn
from .utils.sign_sgd import SignSGD
import copy
import gc
from tqdm import tqdm
from typing import List


def reshape_and_pad_tensor(v, group_size=-1):
    """Reshapes the tensor based on the group size for parameter initialization."""
    if group_size == 0:
        return v.reshape(1, -1)
    if group_size == -1 or v.shape[1] < group_size:
        return v
    if v.shape[1] % group_size == 0:
        v = v.reshape(-1, group_size)
    else:
        pad_len = (v.shape[1] + group_size - 1) // group_size * group_size - v.shape[1]
        v = torch.nn.functional.pad(v, (0, pad_len))
        v = v.reshape(-1, group_size)
    return v


class WrapperLinear(nn.Module):
    """
    A wrapper for linear layers that enables trainable quantization.
    
    This wrapper temporarily replaces a linear layer during the optimization phase,
    introducing learnable parameters for quantization. After training, it can be
    converted back to a QuantizedLinear for efficient inference.
    """
    
    def __init__(
        self,
        orig_layer,
        bits=8,
        group_size=32,
        enable_minmax_tuning=True,
        enable_round_tuning=True,
        device="cpu"
    ):
        super().__init__()
        
        self.orig_layer = orig_layer
        self.bits = bits
        self.group_size = group_size
        self.enable_minmax_tuning = enable_minmax_tuning
        self.enable_round_tuning = enable_round_tuning
        
        # Use the device of the original layer's weight, not the passed device parameter
        self.device = orig_layer.weight.device
        
        # Store original weight for reference (already on correct device)
        self.register_buffer('orig_weight', orig_layer.weight.data.clone())
        if orig_layer.bias is not None:
            self.register_buffer('orig_bias', orig_layer.bias.data.clone())
        else:
            self.orig_bias = None
        
        # Initialize quantization parameters
        self._init_quantization_params()
        
        # Initialize learnable parameters (will be on same device)
        self._init_learnable_params()
        
        # Ensure wrapper is on the same device as original layer
        self.to(self.device)
        
        # Track best parameters during training
        self.best_params = {}
        self.best_loss = float('inf')
    
    def _init_quantization_params(self):
        """Initialize base quantization parameters using RTN."""
        with torch.no_grad():
            # Get RTN quantization parameters
            qweight, scale, zero_point, _ = quantize_rtn(
                self.orig_weight, 
                bits=self.bits, 
                group_size=self.group_size
            )
            
            # Store as buffers (non-trainable)
            self.register_buffer('base_qweight', qweight)
            self.register_buffer('base_scale', scale)
            self.register_buffer('base_zero_point', zero_point)
    
    def _init_learnable_params(self):
        """Initialize learnable parameters for quantization optimization."""
        weight_shape = self.orig_weight.shape
        
        # Learnable rounding parameter (value)
        if self.enable_round_tuning:
            # Initialize with zeros like AutoRound
            value_param = torch.zeros_like(self.orig_weight)
            self.value = nn.Parameter(value_param)
        else:
            self.value = None
        
        # Learnable min/max scale parameters  
        if self.enable_minmax_tuning:
            num_groups = weight_shape[1] // self.group_size
            scale_shape = (weight_shape[0], num_groups)
            
            # Initialize close to 1.0 (no change)
            self.min_scale = nn.Parameter(torch.ones(scale_shape) * 1.0)
            self.max_scale = nn.Parameter(torch.ones(scale_shape) * 1.0)
        else:
            self.min_scale = None
            self.max_scale = None
    
    def _quantize_weight_with_learnable_params(self):
        """Apply quantization using learnable parameters."""
        weight = self.orig_weight.clone()
        
        # AutoRound approach: Apply learnable parameters during quantization process
        out_features, in_features = weight.shape
        num_groups = in_features // self.group_size
        
        # Reshape for group-wise quantization 
        weight_grouped = weight.view(out_features, num_groups, self.group_size)
        
        # Calculate base quantization parameters per group
        max_val = (1 << self.bits) - 1
        w_min = weight_grouped.min(dim=-1, keepdim=True)[0]
        w_max = weight_grouped.max(dim=-1, keepdim=True)[0]
        
        # Apply learnable min/max scale adjustments to the range
        if self.min_scale is not None and self.max_scale is not None:
            min_scale_expanded = self.min_scale.unsqueeze(-1)  # (out_features, num_groups, 1)
            max_scale_expanded = self.max_scale.unsqueeze(-1)  # (out_features, num_groups, 1)
            w_min = w_min * min_scale_expanded
            w_max = w_max * max_scale_expanded
        
        # Calculate scale and zero_point
        scale = (w_max - w_min) / max_val
        scale = torch.clamp(scale, min=1e-8)  # Prevent division by zero
        zero_point = -w_min / scale
        
        # Scale weights to quantized range
        weight_scaled = weight_grouped / scale + zero_point
        
        # AutoRound key innovation: Add learnable rounding perturbation AFTER scaling
        if self.value is not None:
            value_grouped = self.value.view(out_features, num_groups, self.group_size)
            weight_scaled = weight_scaled + value_grouped
        
        # Apply quantization
        weight_q = torch.clamp(torch.round(weight_scaled), 0, max_val)
        
        # Dequantize for forward pass
        weight_dq = scale * (weight_q - zero_point)
        weight_dq = weight_dq.view(out_features, in_features)
        
        # Return with proper shapes for compatibility
        scale_flat = scale.squeeze(-1)      # Shape: (out_features, num_groups)
        zero_point_flat = zero_point.squeeze(-1)  # Shape: (out_features, num_groups)
        
        return weight_dq, weight_q, scale_flat, zero_point_flat
    
    def forward(self, x):
        """Forward pass always using learnable quantization."""
        # Always use learnable parameters (even if they're zeros initially)
        weight_dq, _, _, _ = self._quantize_weight_with_learnable_params()
        
        # Apply linear transformation
        bias = self.orig_bias if self.orig_bias is not None else None
        return F.linear(x, weight_dq, bias)
    
    def update_best_params(self, loss_value):
        """Update best parameters if current loss is better."""
        if loss_value < self.best_loss:
            self.best_loss = loss_value
            self.best_params = {}
            
            if self.value is not None:
                self.best_params['value'] = self.value.data.clone()
            if self.min_scale is not None:
                self.best_params['min_scale'] = self.min_scale.data.clone()
            if self.max_scale is not None:
                self.best_params['max_scale'] = self.max_scale.data.clone()
    
    def apply_best_params(self):
        """Apply the best found parameters."""
        if self.best_params:
            if 'value' in self.best_params and self.value is not None:
                self.value.data.copy_(self.best_params['value'])
            if 'min_scale' in self.best_params and self.min_scale is not None:
                self.min_scale.data.copy_(self.best_params['min_scale'])
            if 'max_scale' in self.best_params and self.max_scale is not None:
                self.max_scale.data.copy_(self.best_params['max_scale'])
    
    def to_quantized_linear(self) -> QuantizedLinear:
        """
        Convert the wrapper to a QuantizedLinear using the optimized parameters.
        This removes all learnable parameters and creates an efficient inference layer.
        """
        with torch.no_grad():
            # Apply best parameters
            self.apply_best_params()
            
            # Get final quantized parameters
            _, weight_q, scale, zero_point = self._quantize_weight_with_learnable_params()
            
            # Create QuantizedLinear instance
            q_layer = QuantizedLinear(
                self.orig_layer.in_features,
                self.orig_layer.out_features, 
                bits=self.bits,
                group_size=self.group_size,
                bias=self.orig_layer.bias is not None
            )
            
            # Use the same quantization approach as your quantize_rtn function
            # Just apply quantization directly to the original weight using optimized parameters
            from .functional import quantize_rtn
            
            # Apply learned adjustments to the original weight
            adjusted_weight = self.orig_weight.clone()
            if self.value is not None:
                adjusted_weight = adjusted_weight + self.value
            
            # Use quantize_rtn with the adjusted weight to get proper packing
            qweight_packed, scale_final, zero_point_final, _ = quantize_rtn(
                adjusted_weight, 
                bits=self.bits, 
                group_size=self.group_size
            )
            
            # Copy parameters (use the final quantization results)
            q_layer.qweight.copy_(qweight_packed)
            q_layer.scale.copy_(scale_final)
            q_layer.zero_point.copy_(zero_point_final)
            
            if self.orig_layer.bias is not None:
                q_layer.bias.copy_(self.orig_layer.bias)
            
            return q_layer


def unified_wrapper(module, enable_minmax_tuning=True, enable_round_tuning=True, bits=8, group_size=32, 
                    device="cpu", ignore_layers=None, module_prefix=""):
    """
    Unified wrapper that handles both individual linear layers and blocks containing linear layers.
    Prevents nested wrapping and provides consistent interface.
    
    Args:
        module: Either a single nn.Linear layer or a block containing linear layers
        enable_minmax_tuning: Enable min/max scale tuning
        enable_round_tuning: Enable rounding value tuning  
        bits: Number of quantization bits
        group_size: Group size for quantization
        device: Device to place tensors on
        ignore_layers: Set of layer names to skip during wrapping
        module_prefix: Prefix to add to layer names when checking ignore_layers
    
    Returns:
        tuple: (wrapped_module, quantized_layer_names) where wrapped_module is the input module 
               (potentially modified) and quantized_layer_names is list of layer names that were wrapped
    """
    ignore_layers = ignore_layers or set()
    quantized_layer_names = []
    
    # Check if already wrapped
    if isinstance(module, WrapperLinear):
        print(f"Module already wrapped, skipping: {module_prefix}")
        return module, []
    
    # Case 1: Single linear layer
    if isinstance(module, nn.Linear):
        if module_prefix in ignore_layers:
            print(f"  Skipping ignored layer: {module_prefix}")
            return module, []
        
        wrapper = WrapperLinear(
            module,
            bits=bits,
            group_size=group_size,
            enable_minmax_tuning=enable_minmax_tuning,
            enable_round_tuning=enable_round_tuning,
            device=device
        )
        return wrapper, ["single_layer"]
    
    # Case 2: Block containing linear layers
    for name, submodule in module.named_modules():
        if isinstance(submodule, nn.Linear):
            # Check if already wrapped
            if isinstance(submodule, WrapperLinear):
                continue
                
            # Construct full layer name for ignore check
            full_layer_name = f"{module_prefix}.{name}" if module_prefix and name else (module_prefix or name)
            
            if full_layer_name in ignore_layers:
                print(f"  Skipping ignored layer: {full_layer_name}")
                continue
            
            # Create wrapper
            wrapper = WrapperLinear(
                submodule,
                bits=bits,
                group_size=group_size,
                enable_minmax_tuning=enable_minmax_tuning,
                enable_round_tuning=enable_round_tuning,
                device=device
            )
            
            # Replace the module
            _set_module(module, name, wrapper)
            quantized_layer_names.append(name)
    
    return module, quantized_layer_names


def unified_unwrapper(module, apply_quantization=True):
    """
    Unified unwrapper that handles both individual WrapperLinear layers and blocks containing them.
    
    Args:
        module: Either a WrapperLinear or a block containing WrapperLinear layers
        apply_quantization: If True, convert to QuantizedLinear. If False, restore original layers.
    
    Returns:
        The unwrapped module (QuantizedLinear, original layer, or modified block)
    """
    # Case 1: Single WrapperLinear
    if isinstance(module, WrapperLinear):
        if apply_quantization:
            return module.to_quantized_linear()
        else:
            return module.orig_layer
    
    # Case 2: Block containing WrapperLinear layers
    for name, submodule in list(module.named_modules()):
        if isinstance(submodule, WrapperLinear):
            if apply_quantization:
                quantized_layer = submodule.to_quantized_linear()
                _set_module(module, name, quantized_layer)
            else:
                _set_module(module, name, submodule.orig_layer)
    
    return module


# Legacy functions for backward compatibility
def wrapper_block(block, enable_minmax_tuning=True, enable_round_tuning=True, bits=8, group_size=32, device="cpu", 
                  ignore_layers=None, block_prefix=""):
    """Legacy function - use unified_wrapper instead."""
    _, quantized_names = unified_wrapper(block, enable_minmax_tuning, enable_round_tuning, bits, group_size, device, ignore_layers, block_prefix)
    unquantized_names = [name for name, module in block.named_modules() if not isinstance(module, WrapperLinear)]
    return quantized_names, unquantized_names


def unwrapper_block(block, apply_quantization=True):
    """Legacy function - use unified_unwrapper instead."""
    unified_unwrapper(block, apply_quantization)

def wrap_model_for_training(model, block_names: List[str], enable_minmax_tuning: bool, 
                           bits: int, group_size: int, ignore_layers: set):
    """
    First phase: Wrap all blocks/layers with WrapperLinear for training.
    
    Args:
        model: The model to wrap
        block_names: List of block/layer names to wrap
        enable_minmax_tuning: Whether to enable min/max scale tuning
        bits: Quantization bits
        group_size: Group size for quantization
        ignore_layers: Set of layer names to ignore
    
    Returns:
        Dictionary mapping block names to their wrapper info
    """
    wrapped_info = {}
    
    for module_name in block_names:
        module = _get_module(model, module_name)
        
        # Wrap the module
        wrapped_module, quantized_names = unified_wrapper(
            module,
            enable_minmax_tuning=enable_minmax_tuning,
            enable_round_tuning=True,
            bits=bits,
            group_size=group_size,
            device=next(module.parameters()).device,
            ignore_layers=ignore_layers,
            module_prefix=module_name
        )
        
        if quantized_names:
            wrapped_info[module_name] = {
                'wrapped_module': wrapped_module,
                'quantized_names': quantized_names,
                'is_single_layer': isinstance(wrapped_module, WrapperLinear),
                'original_module': module
            }
        else:
            print(f"  No layers to wrap in {module_name}")
    
    return wrapped_info
 
def quick_eval_during_training(model, original_model, tokenizer, eval_samples=50):
    """
    Quick evaluation function to check perplexity and KL divergence during training.
    Can be called anytime with the wrapped model.
    
    Args:
        model: Current wrapped model
        original_model: Original model for comparison
        tokenizer: Tokenizer
        eval_samples: Number of samples (small for quick eval)
    
    Returns:
        Dict with perplexity and KL divergence
    """
    from .utils.evaluation import calculate_perplexity, calculate_kl_divergence
    
    model.eval()
    original_model.eval()
    
    print(f"Quick eval ({eval_samples} samples)...")
    
    # Calculate perplexity
    wrapped_ppl = calculate_perplexity(model, tokenizer, eval_samples=eval_samples)
    original_ppl = calculate_perplexity(original_model, tokenizer, eval_samples=eval_samples)
    
    # Calculate KL divergence
    kl_div, token_kl_div = calculate_kl_divergence(original_model, model, tokenizer, eval_samples=eval_samples)
    
    results = {
        'wrapped_ppl': wrapped_ppl,
        'original_ppl': original_ppl,
        'ppl_diff': wrapped_ppl - original_ppl,
        'kl_div': kl_div
    }
    
    print(f"  Original PPL: {original_ppl:.4f} | Wrapped PPL: {wrapped_ppl:.4f} | Diff: {results['ppl_diff']:+.4f}")
    print(f"  KL Divergence: {kl_div:.6f}")
    
    return results


def train_individual_wrapper(module_name: str, wrapped_module, block_inputs: List[torch.Tensor],
                           iters: int, lr: float, batch_size: int, is_single_layer: bool):
    """
    Second phase: Train an individual wrapped module.
    
    Args:
        module_name: Name of the module being trained
        wrapped_module: The wrapped module to train
        block_inputs: Cached inputs for this module
        iters: Number of training epochs
        lr: Learning rate
        batch_size: Training batch size
        is_single_layer: Whether this is a single layer or block
        
    Returns:
        Trained module result or None for blocks
    """
    if not block_inputs:
        print("Warning: No cached inputs for this module, skipping...")
        return None
    
    num_samples = len(block_inputs)
    device = next(wrapped_module.parameters()).device
    
    # Cache original outputs (before training)
    original_outputs = []
    wrapped_module.eval()
    
    with torch.no_grad():
        for inp in tqdm(block_inputs, desc="Caching outputs", leave=False):
            inp = inp.to(device)
            try:
                # Get output from the original module (before wrapping)
                # We need to temporarily unwrap to get clean baseline
                if is_single_layer:
                    orig_out = wrapped_module.orig_layer(inp)
                else:
                    # For blocks, we need the original module - create temporary unwrapped version
                    temp_unwrapped = unified_unwrapper(copy.deepcopy(wrapped_module), apply_quantization=False)
                    orig_out = temp_unwrapped(inp)
                    del temp_unwrapped
                
                # Handle different output formats
                if isinstance(orig_out, tuple):
                    orig_out = orig_out[0]
                
                original_outputs.append(orig_out.detach().cpu())
            except Exception as e:
                print(f"Warning: Failed to cache output: {e}")
                continue
    
    # Collect wrappers for training
    wrappers = []
    if is_single_layer:
        wrappers = [wrapped_module]
    else:
        for name, submodule in wrapped_module.named_modules():
            if isinstance(submodule, WrapperLinear):
                wrappers.append(submodule)
    
    # Setup optimizer parameters
    learnable_params = []
    for i, wrapper in enumerate(wrappers):
        if wrapper.value is not None:
            learnable_params.append(wrapper.value)
        if wrapper.min_scale is not None:
            learnable_params.append(wrapper.min_scale)
        if wrapper.max_scale is not None:
            learnable_params.append(wrapper.max_scale)
    
    if not learnable_params:
        print("Warning: No learnable parameters found")
        return None
    
    optimizer = SignSGD(learnable_params, lr=lr, momentum=0.9)
    
    batch_size = min(batch_size, num_samples)
    steps_per_epoch = (num_samples + batch_size - 1) // batch_size
    total_steps = iters * steps_per_epoch
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps
    )
    
    print(f"Training: {iters} epochs × {steps_per_epoch} steps = {total_steps} total steps")
    
    # Training loop
    wrapped_module.train()
    global_step = 0
    
    for epoch in range(iters):
        epoch_loss = 0.0
        epoch_step_losses = []
        
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            
            optimizer.zero_grad()
            batch_loss = 0.0
            
            # Process mini-batch
            for i in range(batch_start, batch_end):
                inp = block_inputs[i].to(device)
                orig_out = original_outputs[i].to(device)
                
                # Forward pass through wrapped module
                quant_out = wrapped_module(inp)
                
                if isinstance(quant_out, tuple):
                    quant_out = quant_out[0]
                
                # MSE loss
                loss = F.mse_loss(quant_out, orig_out)
                scaled_loss = loss * 1000  # AutoRound scaling
                batch_loss += scaled_loss.item()
                
                scaled_loss.backward()
            
            # Update parameters
            optimizer.step()
            scheduler.step()
            global_step += 1
            
            avg_batch_loss = batch_loss / (batch_end - batch_start)
            epoch_loss += batch_loss
            epoch_step_losses.append(avg_batch_loss)
            
            # Progress reporting
            if global_step % max(1, steps_per_epoch // 4) == 0 or global_step <= 3:
                current_lr = scheduler.get_lr()[0]
                print(f"  Step {global_step}/{total_steps}: loss={avg_batch_loss:.6f}, lr={current_lr:.6f}")
        
        # Epoch summary
        avg_epoch_loss = epoch_loss / num_samples
        epoch_min = min(epoch_step_losses) if epoch_step_losses else 0
        epoch_max = max(epoch_step_losses) if epoch_step_losses else 0
        current_lr = scheduler.get_lr()[0]
        
        print(f"Epoch {epoch + 1}/{iters}: Avg={avg_epoch_loss:.6f}, Min={epoch_min:.6f}, Max={epoch_max:.6f}, LR={current_lr:.6f}")
    
    # Apply best parameters and unwrap
    for wrapper in wrappers:
        wrapper.apply_best_params()
    
    # Convert to quantized
    result = unified_unwrapper(wrapped_module, apply_quantization=True)
    
    # Cleanup
    del original_outputs, learnable_params, optimizer, scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return result if is_single_layer else None


def _get_module(model, module_name: str):
    """Get a module by its dotted name."""
    tokens = module_name.split('.')
    cur_mod = model
    for token in tokens:
        cur_mod = getattr(cur_mod, token)
    return cur_mod


def _check_module_output_similarity(orig_module, wrapped_module, sample_inputs):
    """Check output similarity between original and wrapped module."""
    if not sample_inputs:
        return
        
    orig_module.eval()
    wrapped_module.eval()
    
    total_error = 0.0
    max_error = 0.0
    
    with torch.no_grad():
        for i, inp in enumerate(sample_inputs):
            inp = inp.to(next(orig_module.parameters()).device)
            
            orig_out = orig_module(inp)
            wrapped_out = wrapped_module(inp)
            
            if isinstance(orig_out, tuple):
                orig_out = orig_out[0]
            if isinstance(wrapped_out, tuple):
                wrapped_out = wrapped_out[0]
            
            error = (orig_out - wrapped_out).abs()
            sample_max_error = error.max().item()
            sample_mean_error = error.mean().item()
            
            total_error += sample_mean_error
            max_error = max(max_error, sample_max_error)
            
            if i < 2:  # Show first 2 samples
                print(f"  Sample {i}: max_err={sample_max_error:.6f}, mean_err={sample_mean_error:.6f}")
    
    avg_error = total_error / len(sample_inputs)
    print(f"  Wrapped module error: max={max_error:.6f}, avg={avg_error:.6f}")
    
    if avg_error > 1.0:
        print(f"  WARNING: Large error after wrapping - quantization may be unstable!")


def _set_module(model, submodule_key, module):
    """Helper function to replace a module within a model hierarchy."""
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)