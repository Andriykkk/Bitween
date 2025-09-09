"""
Base wrapper class for block-level modifications with device management.
"""

import torch
import torch.nn as nn
from typing import Optional, Any, Dict
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
            self.wrapped_block.to(x.device)
            
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
                self.wrapped_block.to("cpu")
                
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