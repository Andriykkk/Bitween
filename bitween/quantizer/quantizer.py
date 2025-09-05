import torch
import torch.nn as nn
from ..modules import QuantizedLinear
from ..calib_dataset import get_calibration_dataset
from ..utils.evaluation import calculate_perplexity, calculate_kl_divergence, print_report
import copy
from typing import List, Optional
from .func import _set_module
from .utils.cache_manager import CacheManager

class Bitween:
    """
    A simple quantizer class that takes a model and quantization configuration.
    """
    def __init__(self, model, tokenizer=None, bits=8, group_size=32, iters=1, lr=0.005, enable_minmax_tuning=True, seqlen=2048, 
                 cache_to_disk=False, max_memory_mb=512, ignore_layers=None, batch_size=32):
        self.model = model
        self.tokenizer = tokenizer
        self.bits = bits
        self.group_size = group_size
        
        # Training parameters for trainable quantization
        self.iters = iters  # Number of epochs (full passes through all samples) per block
        self.lr = lr        # Learning rate for optimization
        self.enable_minmax_tuning = enable_minmax_tuning  # Enable min/max scale tuning
        self.seqlen = seqlen  # Sequence length for calibration data
        
        # Memory management parameters
        self.cache_to_disk = cache_to_disk  # Save cache tensors to filesystem
        self.max_memory_mb = max_memory_mb  # Maximum memory for caching (MB)
        
        # Layers to ignore during quantization
        self.ignore_layers = set(ignore_layers) if ignore_layers else set()
        self.batch_size = batch_size

        assert group_size > 0, "Group size must be greater than 0."

    def quantize(self, evaluate_perplexity=False, eval_samples=100, rtn=False, trainable=False, calib_dataset="pile-10k", nsamples=None, 
                 cache_to_disk=None, max_memory_mb=None, ignore_layers=None, batch_size=None, **eval_kwargs):
        """
        Quantizes the linear layers of the model.

        Args:
            evaluate_perplexity (bool): If True, run a full performance evaluation.
            eval_samples (int): Number of samples to use for evaluation.
            rtn (bool): If True, use RTN quantization (fast, lower quality).
            trainable (bool): If True, use trainable quantization (slower, higher quality).
            calib_dataset (str): Dataset name for calibration ('pile-10k', etc.).
            nsamples (int): Number of calibration samples to use for training. If None, uses all available samples.
            cache_to_disk (bool): Override instance setting for disk caching. If None, uses instance setting.
            max_memory_mb (int): Override instance setting for memory limit. If None, uses instance setting.
            ignore_layers (list): List of layer names to skip during quantization (e.g., ['lm_head', 'embed_tokens']).
            batch_size (int): Override batch size for training mini-batches. If None, uses instance setting.
            **eval_kwargs: Additional arguments for the evaluation functions.
        
        Returns:
            Quantized model using QuantizedLinear layers with optimized parameters.
        """
        # Handle override parameters - temporarily update instance variables
        original_ignore_layers = self.ignore_layers
        
        if ignore_layers is not None:
            self.ignore_layers = set(ignore_layers)
        if batch_size is not None:
            self.batch_size = batch_size
            
        original_ppl = None
        if evaluate_perplexity and self.tokenizer is not None and eval_samples > 0:
            if self.tokenizer is None:
                raise ValueError("Tokenizer must be provided for evaluation.")
            print("\n--- Evaluating original model ---")
            original_ppl = calculate_perplexity(self.model, self.tokenizer, eval_samples=eval_samples, **eval_kwargs)

        print("\n--- Starting quantization ---")

        if not trainable:        
            # Create a deepcopy for quantization to keep the original model intact for KL-divergence
            quantized_model = copy.deepcopy(self.model)
            
            # Find and replace all linear layers
            for name, module in quantized_model.named_modules():
                if isinstance(module, nn.Linear):
                    if name in self.ignore_layers:
                        print(f"  - Skipping ignored layer: {name}")
                        continue
                        
                    print(f"  - Quantizing layer: {name}")
                    
                    q_module = QuantizedLinear.from_float(
                        module, self.bits, self.group_size
                    )
                    
                    _set_module(quantized_model, name, q_module)

        if trainable and not rtn:
            # Override cache settings if specified
            if cache_to_disk is not None:
                self.cache_to_disk = cache_to_disk
            if max_memory_mb is not None:
                self.max_memory_mb = max_memory_mb
                
            print(f"Trainable mode: lr={self.lr}, minmax_tuning={self.enable_minmax_tuning}")
            print(f"Cache mode: {'disk' if self.cache_to_disk else 'memory'} (max: {self.max_memory_mb}MB)")
            # Trainable quantization: Use calibration data to optimize quantization parameters
            print(f"Using trainable quantization with {calib_dataset} dataset")
            quantized_model = self._trainable_quantize(calib_dataset, nsamples, self.batch_size)
                
        print("--- Quantization complete ---")

        if evaluate_perplexity and self.tokenizer is not None and eval_samples > 0:
            print("\n--- Evaluating quantized model ---")
            quantized_ppl = calculate_perplexity(quantized_model, self.tokenizer, eval_samples=eval_samples, **eval_kwargs)
            
            print("\n--- Calculating KL-Divergence ---")
            kl_div, token_kl_div = calculate_kl_divergence(self.model, quantized_model, self.tokenizer, eval_samples=eval_samples, **eval_kwargs)
            
            print_report(original_ppl, quantized_ppl, kl_div, token_kl_div)

        # Restore original settings
        self.ignore_layers = original_ignore_layers
        return quantized_model

    def _trainable_quantize(self, calib_dataset: str, nsamples: Optional[int], batch_size: int):
        """
        Performs trainable quantization using calibration dataset.
        
        Args:
            calib_dataset (str): Name of calibration dataset
            nsamples (Optional[int]): Number of calibration samples. If None, uses all available.
            batch_size (int): Training batch size
            
        Returns:
            Quantized model with optimized parameters
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for trainable quantization")
        
        # Create a copy of the model for quantization
        quantized_model = copy.deepcopy(self.model)
        
        # Step 1: Get block names (detect model architecture blocks)
        block_names = self._get_block_names(quantized_model)
        print(f"Detected {len(block_names)} blocks: {block_names}")
        
        # Step 2: Load calibration dataset 
        print(f"Loading calibration dataset: {calib_dataset}")
        dataset_nsamples = nsamples if nsamples is not None else 10000
        calib_data = get_calibration_dataset(
            dataset_name=calib_dataset,
            tokenizer=self.tokenizer,
            seqlen=self.seqlen,
            nsamples=dataset_nsamples
        )
        
        actual_nsamples = len(calib_data)
        print(f"Using {actual_nsamples} calibration samples for training")
        
        # Step 3: Cache inputs and quantize each module
        cached_inputs = CacheManager.cache_block_inputs(
            quantized_model, calib_data, block_names, actual_nsamples,
            cache_to_disk=self.cache_to_disk, max_memory_mb=self.max_memory_mb
        )
        
        try:
            for module_name in block_names:
                print(f"\n--- Quantizing: {module_name} ---")
                module = self._get_module(quantized_model, module_name)
                
                # Load cached inputs for this module
                block_inputs = CacheManager.load_block_cache(cached_inputs, module_name, cache_to_disk=self.cache_to_disk)
                
                if not block_inputs:
                    print(f"Warning: No cached inputs found for {module_name}, skipping...")
                    continue
                
                # Quantize the module using unified approach
                result = self._quantize_block(module, block_inputs, module_name, batch_size)
                
                # If it's a single layer, we need to replace it in the model
                if isinstance(module, nn.Linear) and result is not None:
                    _set_module(quantized_model, module_name, result)
                    print(f"Replaced {module_name} with quantized version")
                
                # Clean up cache for this module to free memory
                CacheManager.cleanup_block_cache(cached_inputs, module_name, cache_to_disk=self.cache_to_disk)
        
        finally:
            # Always clean up all cache files when done
            CacheManager.cleanup_all_cache(cached_inputs, cache_to_disk=self.cache_to_disk)
        
        return quantized_model
    
    
    def _get_block_names(self, model) -> List[str]:
        """
        Detect transformer blocks and standalone layers in the model architecture.
        
        This function identifies:
        1. Main transformer blocks that should be quantized together
        2. Standalone linear layers that are not part of any block
        
        Returns:
            List of block names and standalone layer names
        """
        block_names = []
        all_linear_layers = set()
        layers_in_blocks = set()
        
        # Collect all linear layers first, excluding ignored layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name not in self.ignore_layers:
                all_linear_layers.add(name)
        
        # Common patterns for transformer blocks
        patterns = [
            'layers',      # LLaMA, Mistral: model.layers.0, model.layers.1, ...
            'h',           # GPT: transformer.h.0, transformer.h.1, ...
            'blocks',      # Some models: model.blocks.0, model.blocks.1, ...
            'decoder',     # T5: decoder.block.0, decoder.block.1, ...
        ]
        
        # Find transformer blocks
        for name, module in model.named_modules():
            # Look for numbered blocks (e.g., layers.0, h.1, blocks.2, etc.)
            for pattern in patterns:
                if pattern in name and any(c.isdigit() for c in name):
                    # Extract the full block path (e.g., "model.layers.0")
                    parts = name.split('.')
                    for i, part in enumerate(parts):
                        if pattern in part and i + 1 < len(parts) and parts[i + 1].isdigit():
                            block_path = '.'.join(parts[:i + 2])
                            if block_path not in block_names:
                                block_names.append(block_path)
                                
                                # Mark all linear layers in this block as "covered"
                                block_module = self._get_module(model, block_path)
                                for sub_name, sub_module in block_module.named_modules():
                                    if isinstance(sub_module, nn.Linear):
                                        full_layer_name = f"{block_path}.{sub_name}" if sub_name else block_path
                                        if full_layer_name not in self.ignore_layers:
                                            layers_in_blocks.add(full_layer_name)
                            break
                    break
        
        # Filter out blocks that contain only ignored layers
        filtered_block_names = []
        for block_name in block_names:
            block_module = self._get_module(model, block_name)
            has_quantizable_layers = False
            
            for sub_name, sub_module in block_module.named_modules():
                if isinstance(sub_module, nn.Linear):
                    full_layer_name = f"{block_name}.{sub_name}" if sub_name else block_name
                    if full_layer_name not in self.ignore_layers:
                        has_quantizable_layers = True
                        break
            
            if has_quantizable_layers:
                filtered_block_names.append(block_name)
            else:
                print(f"Skipping block '{block_name}' - all linear layers are in ignore list")
        
        block_names = filtered_block_names
        
        # Sort block names to ensure consistent processing order
        block_names.sort()
        
        # Find standalone linear layers (not covered by any block)
        standalone_layers = all_linear_layers - layers_in_blocks
        standalone_layers = sorted(list(standalone_layers))
        
        # Add standalone layers to the processing list
        if standalone_layers:
            print(f"Found {len(standalone_layers)} standalone linear layers:")
            for layer_name in standalone_layers:
                print(f"  - {layer_name}")
                block_names.append(layer_name)
        
        # Print ignored layers summary
        if self.ignore_layers:
            print(f"Ignoring {len(self.ignore_layers)} layers from quantization:")
            for ignored_layer in sorted(self.ignore_layers):
                print(f"  - {ignored_layer}")
        
        if block_names:
            transformer_blocks = [name for name in block_names if name not in standalone_layers]
            print(f"Found {len(transformer_blocks)} transformer blocks and {len(standalone_layers)} standalone layers")
        else:
            print("Warning: No transformer blocks or linear layers detected.")
        
        return block_names
    
    
    def _quantize_block(self, module, block_inputs: List[torch.Tensor], module_name: str, batch_size: int):
        """
        Simplified quantization function that delegates training to wrapper.py.
        
        Args:
            module: Module to quantize (block or individual layer)
            block_inputs: List of ALL input tensors cached for this module
            module_name: Name of the module for logging and ignore checking
            batch_size: Training batch size
        """
        from ..wrapper import train_wrapped_module
        
        # Delegate all training logic to wrapper.py
        return train_wrapped_module(
            module=module,
            block_inputs=block_inputs,
            module_name=module_name,
            iters=self.iters,
            lr=self.lr,
            batch_size=batch_size,
            enable_minmax_tuning=self.enable_minmax_tuning,
            bits=self.bits,
            group_size=self.group_size,
            ignore_layers=self.ignore_layers
        )
    
    
    def _get_module(self, model, module_name: str):
        """Get a module by its dotted name."""
        return CacheManager._get_module(model, module_name)

