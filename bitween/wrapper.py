import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import QuantizedLinear
from .functional import quantize_rtn, dequantize_rtn
import copy


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
            # Initialize with small random values around zero
            value_param = torch.zeros_like(self.orig_weight) * 0.01
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
        
        # Apply learnable rounding adjustment
        if self.value is not None:
            weight = weight + self.value
        
        # Use the base quantization approach with learnable adjustments
        out_features, in_features = weight.shape
        num_groups = in_features // self.group_size
        
        # Reshape for group-wise quantization
        weight_grouped = weight.view(out_features, num_groups, self.group_size)
        
        # Calculate quantization parameters per group
        max_val = (1 << self.bits) - 1
        w_min = weight_grouped.min(dim=-1, keepdim=True)[0]  # Shape: (out_features, num_groups, 1)
        w_max = weight_grouped.max(dim=-1, keepdim=True)[0]  # Shape: (out_features, num_groups, 1)
        
        # Base scale and zero_point calculation
        scale = (w_max - w_min) / max_val  # Shape: (out_features, num_groups, 1)
        zero_point = -w_min / scale        # Shape: (out_features, num_groups, 1)
        
        # Apply learnable min/max scaling adjustments
        if self.min_scale is not None and self.max_scale is not None:
            # min_scale and max_scale have shape (out_features, num_groups)
            # We need to add a dimension for broadcasting
            scale_adjustment = self.min_scale.unsqueeze(-1) * self.max_scale.unsqueeze(-1)  # (out_features, num_groups, 1)
            scale = scale * scale_adjustment
        
        # Apply quantization
        weight_q = torch.clamp(
            torch.round(weight_grouped / scale + zero_point),
            0, max_val
        )
        
        # Dequantize for forward pass
        weight_dq = scale * (weight_q - zero_point)
        weight_dq = weight_dq.view(out_features, in_features)
        
        # Return with proper shapes for compatibility
        scale_flat = scale.squeeze(-1)      # Shape: (out_features, num_groups)
        zero_point_flat = zero_point.squeeze(-1)  # Shape: (out_features, num_groups)
        
        return weight_dq, weight_q, scale_flat, zero_point_flat
    
    def forward(self, x):
        """Forward pass with learnable quantization."""
        # Get quantized weight using learnable parameters
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


def _set_module(model, submodule_key, module):
    """Helper function to replace a module within a model hierarchy."""
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)