"""
Base wrapper class for block-level modifications with device management.
"""

import torch
import torch.nn as nn
from typing import Optional, Any, Dict, List
import abc

class BaseBlockWrapper(nn.Module):
    """
    Base wrapper class that provides:
    1. CPU/GPU memory management - keeps block on CPU, loads to GPU only during forward pass
    2. Clean inheritance interface for noise injection, quantization, etc.
    3. Enable/disable functionality 
    4. Transparent drop-in replacement for original blocks
    """
    
    def __init__(
        self, 
        wrapped_block: nn.Module, 
        storage_device: str = "cpu",
        block_name: str = None
    ):
        """
        Initialize base wrapper.
        
        Args:
            wrapped_block: The original block/module to wrap
            storage_device: Device to store block when not in use ("cpu" or "cuda")
            block_name: Optional name for logging/debugging
        """
        super().__init__()
        
        self.wrapped_block = wrapped_block
        self.storage_device = storage_device
        self.block_name = block_name or "unnamed_block"
        self.enabled = True
        
        # Move block to storage device initially
        if storage_device == "cpu":
            self.wrapped_block.to("cpu")
            
        # Store original device for restoration
        self._original_device = next(wrapped_block.parameters()).device
        
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass with device management and modification hooks.
        
        Args:
            x: Input tensor
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments (attention_mask, etc.)
            
        Returns:
            Output tensor
        """
        if not self.enabled:
            # If disabled, act as passthrough
            return self.wrapped_block(x, *args, **kwargs)
            
        # Device management: move block to GPU if stored on CPU
        if self.storage_device == "cpu":
            # Move the entire module to the target device
            # This should handle all parameters and buffers automatically
            self.wrapped_block = self.wrapped_block.to(x.device)
            
            # Double check that quantized layers have their buffers on the right device
            for module in self.wrapped_block.modules():
                if hasattr(module, 'qweight'):
                    # Verify device alignment - if misaligned, fix it
                    if module.qweight.device != x.device:
                        module.qweight = module.qweight.to(x.device)
                    if module.scale.device != x.device:
                        module.scale = module.scale.to(x.device)
                    if module.zero_point.device != x.device:
                        module.zero_point = module.zero_point.to(x.device)
                    if module.bias is not None and module.bias.device != x.device:
                        module.bias = module.bias.to(x.device)
            
        try:
            # Pre-forward hook for modifications (noise, quantization, etc.)
            x = self._pre_forward_hook(x)
            
            # Actual forward pass with all arguments
            output = self.wrapped_block(x, *args, **kwargs)
            
            # Post-forward hook for cleanup/modifications
            output = self._post_forward_hook(output)
            
        finally:
            # Always return block to storage device to save memory
            if self.storage_device == "cpu":
                # Move the entire module back to CPU
                self.wrapped_block = self.wrapped_block.to("cpu")
                
        return output
        
    def _pre_forward_hook(self, x: torch.Tensor) -> torch.Tensor:
        """
        Hook called before forward pass. Override in subclasses.
        
        Args:
            x: Input tensor
            
        Returns:
            Potentially modified input tensor
        """
        return x
        
    def _post_forward_hook(self, output: torch.Tensor) -> torch.Tensor:
        """
        Hook called after forward pass. Override in subclasses.
        
        Args:
            output: Output tensor from wrapped block
            
        Returns:
            Potentially modified output tensor
        """
        return output
        
    def enable(self):
        """Enable wrapper functionality."""
        self.enabled = True
        
    def disable(self):
        """Disable wrapper functionality (acts as passthrough)."""
        self.enabled = False
        
    def to_device(self, device: str):
        """
        Manually control wrapper storage device.
        
        Args:
            device: Target device ("cpu" or "cuda")
        """
        self.storage_device = device
        if device != "cpu":
            self.wrapped_block.to(device)
            
    def get_wrapped_block(self) -> nn.Module:
        """Get the original wrapped block."""
        return self.wrapped_block
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.block_name}, storage={self.storage_device}, enabled={self.enabled})"


class NoiseWrapper(BaseBlockWrapper):
    """
    Wrapper that adds Gaussian noise to block weights during forward pass.
    Used for importance analysis.
    """
    
    def __init__(
        self,
        wrapped_block: nn.Module,
        noise_std: float = 0.1, 
        seed: Optional[int] = None,
        storage_device: str = "cpu",
        block_name: str = None
    ):
        """
        Initialize noise wrapper.
        
        Args:
            wrapped_block: Block to wrap
            noise_std: Standard deviation of Gaussian noise (relative to weight std)
            seed: Random seed for deterministic noise
            storage_device: Device management setting
            block_name: Name for identification
        """
        super().__init__(wrapped_block, storage_device, block_name)
        
        self.noise_std = noise_std
        self.seed = seed
        self.noise_generator = torch.Generator()
        if seed is not None:
            self.noise_generator.manual_seed(seed)
            
        # Cache for original weights (for restoration if needed)
        self._original_weights = None
        self._noise_applied = False
        
    def _pre_forward_hook(self, x: torch.Tensor) -> torch.Tensor:
        """Add noise to weights before forward pass."""
        if self.enabled:
            self._add_noise_to_weights(x.device)
        return x
        
    def _post_forward_hook(self, output: torch.Tensor) -> torch.Tensor:
        """Remove noise from weights after forward pass."""
        if self.enabled and self._noise_applied:
            self._remove_noise_from_weights()
        return output
        
    def _add_noise_to_weights(self, device: str):
        """Add Gaussian noise to all linear layer weights in the block."""
        if self._noise_applied:
            return
            
        for name, param in self.wrapped_block.named_parameters():
            if 'weight' in name and param.requires_grad:
                # Calculate noise proportional to parameter std
                param_std = param.data.std()
                noise_scale = self.noise_std * param_std
                
                # Create generator on the same device as parameter
                if param.device.type == 'cuda':
                    generator = torch.Generator(device=param.device)
                    if self.seed is not None:
                        generator.manual_seed(self.seed)
                else:
                    # Use CPU generator for CPU tensors
                    generator = self.noise_generator
                
                # Generate deterministic noise on the same device as parameter
                noise = torch.randn(
                    param.shape, 
                    generator=generator,
                    device=param.device,
                    dtype=param.dtype
                ) * noise_scale
                
                # Add noise to parameter
                param.data += noise
                
        self._noise_applied = True
        
    def _remove_noise_from_weights(self):
        """Remove noise by regenerating and subtracting it."""
        if not self._noise_applied:
            return
            
        for name, param in self.wrapped_block.named_parameters():
            if 'weight' in name and param.requires_grad:
                # Regenerate same noise
                param_std = param.data.std()  # This is noisy std, but close enough
                noise_scale = self.noise_std * param_std
                
                # Create generator on the same device as parameter (same logic as add)
                if param.device.type == 'cuda':
                    generator = torch.Generator(device=param.device)
                    if self.seed is not None:
                        generator.manual_seed(self.seed)
                else:
                    # Use CPU generator for CPU tensors
                    generator = self.noise_generator
                    if self.seed is not None:
                        generator.manual_seed(self.seed)
                
                noise = torch.randn(
                    param.shape,
                    generator=generator, 
                    device=param.device,
                    dtype=param.dtype
                ) * noise_scale
                
                # Subtract noise to restore original weights
                param.data -= noise
                
        self._noise_applied = False
        
    def set_noise_seed(self, seed: int):
        """Change the noise seed."""
        self.seed = seed
        self.noise_generator.manual_seed(seed)


class RTNWrapper(BaseBlockWrapper):
    """
    Wrapper that applies RTN quantization to all linear layers in a block.
    Used for RTN sensitivity analysis.
    """
    
    def __init__(
        self,
        wrapped_block: nn.Module,
        bits: int = 8,
        group_size: int = 32,
        storage_device: str = "cpu",
        block_name: str = None
    ):
        """
        Initialize RTN wrapper.
        
        Args:
            wrapped_block: Block to wrap
            bits: Quantization bits (8, 4, or 2)
            group_size: Group size for quantization
            storage_device: Device management setting
            block_name: Name for identification
        """
        super().__init__(wrapped_block, storage_device, block_name)
        
        self.bits = bits
        self.group_size = group_size
        
        # Store original linear layers for restoration
        self.original_layers = {}
        self.quantized_layers = {}
        self.quantization_applied = False
        
    def _pre_forward_hook(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RTN quantization before forward pass."""
        if self.enabled and not self.quantization_applied:
            # Ensure we're applying quantization when the block is on the right device
            target_device = x.device
            self._apply_rtn_quantization(target_device)
        return x
        
    def _post_forward_hook(self, output: torch.Tensor) -> torch.Tensor:
        """Clean up after forward pass."""
        return output
        
    def _apply_rtn_quantization(self, target_device=None):
        """Apply RTN quantization to all linear layers in the block."""
        if self.quantization_applied:
            return
            
        from ..modules import QuantizedLinear
        
        # Find and quantize all linear layers
        for name, module in self.wrapped_block.named_modules():
            if isinstance(module, nn.Linear):
                # Store original layer
                full_name = f"{name}" if name else "root"
                self.original_layers[full_name] = module
                
                # Create quantized version - ensure it's created on the right device
                device_for_quantization = target_device if target_device is not None else module.weight.device
                
                # Temporarily move module to target device if needed for quantization
                original_device = module.weight.device
                if device_for_quantization != original_device:
                    module = module.to(device_for_quantization)
                
                quantized_layer = QuantizedLinear.from_float(
                    module, 
                    bits=self.bits, 
                    group_size=self.group_size
                )
                
                # Ensure quantized layer is on the target device
                quantized_layer = quantized_layer.to(device_for_quantization)
                
                # Replace the layer
                self._set_submodule(self.wrapped_block, name, quantized_layer)
                self.quantized_layers[full_name] = quantized_layer
                
        self.quantization_applied = True
        
    def _remove_rtn_quantization(self):
        """Restore original linear layers."""
        if not self.quantization_applied:
            return
            
        # Restore all original layers
        for name, original_layer in self.original_layers.items():
            if name == "root":
                # Handle root level replacement if needed
                continue
            else:
                self._set_submodule(self.wrapped_block, name, original_layer)
                
        self.quantization_applied = False
        
    def _set_submodule(self, parent_module, module_name, new_module):
        """Set submodule by name."""
        if not module_name:  # Root level
            return
            
        parts = module_name.split('.')
        current = parent_module
        
        # Navigate to parent
        for part in parts[:-1]:
            current = getattr(current, part)
            
        # Set the new module
        setattr(current, parts[-1], new_module)
        
    def enable(self):
        """Enable RTN quantization."""
        super().enable()
        if not self.quantization_applied:
            self._apply_rtn_quantization()
            
    def disable(self):
        """Disable RTN quantization and restore original layers."""
        super().disable()
        self._remove_rtn_quantization()
        
    def cleanup(self):
        """Clean up and restore original state."""
        self._remove_rtn_quantization()
        self.original_layers.clear()
        self.quantized_layers.clear()


class BlockCacheWrapper(BaseBlockWrapper):
    """
    Simple wrapper that captures block inputs and outputs for training.
    
    No complex yielding - just captures data when enabled.
    """
    
    def __init__(
        self,
        wrapped_block: nn.Module,
        block_name: str,
        storage_device: str = "cpu"
    ):
        """
        Initialize cache wrapper.
        
        Args:
            wrapped_block: Block to wrap
            block_name: Name for identification  
            storage_device: Device management setting
        """
        super().__init__(wrapped_block, storage_device, block_name)
        
        self.cached_inputs = []   # List of input dicts for all samples
        self.cached_outputs = []  # List of output tensors for all samples
        self.capture_enabled = False  # Start disabled
        
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass with optional caching.
        """
        # Device management
        if self.storage_device == "cpu":
            self.wrapped_block = self.wrapped_block.to(x.device)
            
        try:
            # Run forward pass
            output = self.wrapped_block(x, *args, **kwargs)
            
            # Cache data if enabled
            if self.capture_enabled:
                # Cache input for this sample
                input_cache = {
                    'input': x.clone().detach().cpu(),  # Store on CPU to save GPU memory
                    'args': [arg.clone().detach().cpu() if torch.is_tensor(arg) else arg for arg in args],
                    'kwargs': {k: v.clone().detach().cpu() if torch.is_tensor(v) else v for k, v in kwargs.items()}
                }
                
                # Cache output for this sample - handle different output types
                if torch.is_tensor(output):
                    output_cache = output.clone().detach().cpu()
                elif isinstance(output, (tuple, list)):
                    # Handle tuple/list outputs
                    output_cache = type(output)(
                        item.clone().detach().cpu() if torch.is_tensor(item) else item 
                        for item in output
                    )
                else:
                    # For other types, store as-is
                    output_cache = output
                
                # Store cached data
                self.cached_inputs.append(input_cache)
                self.cached_outputs.append(output_cache)
                
            return output
            
        finally:
            # Return block to storage device
            if self.storage_device == "cpu":
                self.wrapped_block = self.wrapped_block.to("cpu")
                
    def get_all_cached_data(self):
        """
        Get all cached input/output pairs.
        
        Returns:
            List of (input_dict, output_tensor) pairs
        """
        return list(zip(self.cached_inputs, self.cached_outputs))
        
    def get_cached_count(self):
        """Get number of cached samples."""
        return len(self.cached_inputs)
        
    def clear_cache(self):
        """Clear cached data to free memory."""
        self.cached_inputs.clear()
        self.cached_outputs.clear()
        
    def enable_capture(self):
        """Enable activation capture."""
        self.capture_enabled = True
        
    def disable_capture(self):
        """Disable activation capture for normal execution."""
        self.capture_enabled = False
        


class BlockTrainingManager:
    """
    Simple manager for block-by-block training with input/output capture.
    
    Approach:
    1. Wrap all blocks with cache wrappers
    2. For each block, enable its capture and run forward passes
    3. Train the block on captured data
    4. Move to next block
    """
    
    def __init__(self, model: nn.Module, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.wrapped_blocks = {}
        self.block_order = []  # Order to process blocks
        
    def wrap_blocks_for_training(self, block_names: List[str]) -> None:
        """
        Wrap specified blocks with cache wrappers.
        
        Args:
            block_names: List of block names to wrap
        """
        from .utils import get_module_by_name, set_module_by_name
        
        print(f"Wrapping {len(block_names)} blocks for training...")
        
        for block_name in block_names:
            # Get original block
            original_block = get_module_by_name(self.model, block_name)
            
            # Create cache wrapper
            wrapper = BlockCacheWrapper(
                wrapped_block=original_block,
                block_name=block_name,
                storage_device="cpu"  # Store cached data on CPU
            )
            
            # Replace block with wrapper
            set_module_by_name(self.model, block_name, wrapper)
            self.wrapped_blocks[block_name] = wrapper
            
        self.block_order = block_names
        print(f"Successfully wrapped blocks: {list(self.wrapped_blocks.keys())}")
        
    def capture_block_data(self, block_name: str, dataset_samples, max_samples=None):
        """
        Capture input/output data for a specific block.
        
        Args:
            block_name: Name of block to capture data for
            dataset_samples: Input samples to run through model
            max_samples: Maximum samples to process
            
        Returns:
            Number of samples captured
        """
        if block_name not in self.wrapped_blocks:
            raise ValueError(f"Block {block_name} is not wrapped")
            
        wrapper = self.wrapped_blocks[block_name]
        
        # Clear any existing cache
        wrapper.clear_cache()
        
        # Enable capture for this block only
        wrapper.enable_capture()
        
        print(f"Capturing data for {block_name}...")
        
        self.model.eval()
        samples_processed = 0
        
        with torch.no_grad():
            for sample in dataset_samples:
                if max_samples and samples_processed >= max_samples:
                    break
                    
                # Move sample to device
                device = next(self.model.parameters()).device
                gpu_sample = {}
                
                for key, value in sample.items():
                    if torch.is_tensor(value):
                        # Ensure tensor has correct shape and device
                        tensor = value.to(device, non_blocking=True)
                        # Add batch dimension if missing
                        if tensor.dim() == 1:
                            tensor = tensor.unsqueeze(0)
                        gpu_sample[key] = tensor
                    else:
                        gpu_sample[key] = value
                
                # Run forward pass - wrapper will capture data
                try:
                    _ = self.model(**gpu_sample)
                    samples_processed += 1
                except Exception as e:
                    print(f"Forward pass failed for sample {samples_processed}: {e}")
                    continue
        
        # Disable capture
        wrapper.disable_capture()
        
        captured_count = wrapper.get_cached_count()
        print(f"Captured {captured_count} samples for {block_name}")
        
        return captured_count
        
    def process_all_blocks_sequentially(self, dataset_samples, max_samples=None):
        """
        Process all blocks sequentially, capturing data for each.
        
        Args:
            dataset_samples: Input samples
            max_samples: Maximum samples per block
            
        Yields:
            (block_name, cached_data) for each block
        """
        for block_name in self.block_order:
            print(f"\n=== Processing Block: {block_name} ===")
            
            # Capture data for this block
            captured_count = self.capture_block_data(block_name, dataset_samples, max_samples)
            
            if captured_count > 0:
                # Get cached data
                cached_data = self.get_block_cached_data(block_name)
                yield block_name, cached_data
            else:
                print(f"No data captured for {block_name}, skipping")
        
    def get_block_cached_data(self, block_name: str):
        """
        Get all cached data for a specific block.
        
        Args:
            block_name: Name of block
            
        Returns:
            List of (input_dict, output_tensor) pairs
        """
        if block_name not in self.wrapped_blocks:
            raise ValueError(f"Block {block_name} is not wrapped")
            
        return self.wrapped_blocks[block_name].get_all_cached_data()
        
    def clear_block_cache(self, block_name: str):
        """Clear cached data for specific block to free memory."""
        if block_name in self.wrapped_blocks:
            self.wrapped_blocks[block_name].clear_cache()
            print(f"Cleared cache for {block_name}")
            
    def cleanup(self):
        """
        Remove all wrappers and restore original blocks.
        """
        from .utils import set_module_by_name
        
        print("Cleaning up block training wrappers...")
        
        for block_name, wrapper in self.wrapped_blocks.items():
            # Get original block
            original_block = wrapper.get_wrapped_block()
            
            # Restore original block
            set_module_by_name(self.model, block_name, original_block)
            
        self.wrapped_blocks.clear()
        self.block_order.clear()
        
        print("Cleanup complete - all wrappers removed")