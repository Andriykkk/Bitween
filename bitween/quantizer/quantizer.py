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

        original_ppl = None
        if evaluate_perplexity and self.tokenizer is not None and eval_samples > 0:
            if self.tokenizer is None:
                raise ValueError("Tokenizer must be provided for evaluation.")
            original_ppl = calculate_perplexity(self.model, self.tokenizer, verbose=False, eval_samples=eval_samples, **eval_kwargs)

        if evaluate_perplexity and self.tokenizer is not None and eval_samples > 0:
            print("\n--- Evaluating quantized model ---")
            quantized_ppl = calculate_perplexity(quantized_model, self.tokenizer, verbose=False, eval_samples=eval_samples, **eval_kwargs)
            
            print("\n--- Calculating KL-Divergence ---")
            kl_div, token_kl_div = calculate_kl_divergence(self.model, quantized_model, self.tokenizer, eval_samples=eval_samples, **eval_kwargs)
            
            print_report(original_ppl, quantized_ppl, kl_div, token_kl_div)

        # Restore original settings
        self.ignore_layers = original_ignore_layers
        return quantized_model

    def _trainable_quantize(self, calib_dataset: str, nsamples: Optional[int], batch_size: int):
        """
        Performs trainable quantization using new wrap-then-train approach.
        
        Args:
            calib_dataset (str): Name of calibration dataset
            nsamples (Optional[int]): Number of calibration samples. If None, uses all available.
            batch_size (int): Training batch size
            
        Returns:
            Quantized model with optimized parameters
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for trainable quantization")
        
        # Keep reference to original model for comparison
        self._original_model = self.model
        
        # Create a copy of the model for quantization
        quantized_model = copy.deepcopy(self.model)
        
        # Step 1: Get block names (detect model architecture blocks)  
        from ..gradual.utils import get_transformer_block_names
        block_names = get_transformer_block_names(quantized_model, self.ignore_layers)
        print(f"Detected {len(block_names)} blocks: {block_names}")
        
        # Step 2: Load calibration dataset 
        dataset_nsamples = nsamples if nsamples is not None else 10000
        calib_data = get_calibration_dataset(
            dataset_name=calib_dataset,
            tokenizer=self.tokenizer,
            seqlen=self.seqlen,
            nsamples=dataset_nsamples
        )
        
        actual_nsamples = len(calib_data)
        
        # Step 3: Cache inputs for all modules
        cached_inputs = CacheManager.cache_block_inputs(
            quantized_model, calib_data, block_names, actual_nsamples,
            cache_to_disk=self.cache_to_disk, max_memory_mb=self.max_memory_mb
        )
        
        try:
            # Step 4: PHASE 1 - Wrap all modules
            from ..wrapper import wrap_model_for_training
            wrapped_info = wrap_model_for_training(
                model=quantized_model,
                block_names=block_names,
                enable_minmax_tuning=self.enable_minmax_tuning,
                bits=self.bits,
                group_size=self.group_size,
                ignore_layers=self.ignore_layers
            )
            
            # Store wrapped info for user access
            self._wrapped_info = wrapped_info
            self._cached_inputs = cached_inputs
            
            self.model = quantized_model
            
            self.train_all()
            
            # Return the trained model
            return quantized_model
        
        except Exception as e:
            # Always clean up on error
            CacheManager.cleanup_all_cache(cached_inputs, cache_to_disk=self.cache_to_disk)
            raise e
    
    def quick_eval(self, eval_samples=50, verbose=False, **eval_kwargs):
        """Quick evaluation of wrapped model vs original."""
        if not hasattr(self, '_wrapped_info'):
            raise ValueError("Model not wrapped yet. Call quantize() first.")
            
        from ..wrapper import quick_eval_during_training
        return quick_eval_during_training(
            model=self.model, 
            original_model=self._original_model,
            tokenizer=self.tokenizer,
            eval_samples=eval_samples,
            verbose=verbose
        )
    
    def train_module(self, module_name: str, additional_iters: Optional[int] = None, auto_eval=True, eval_samples=50):
        """Train a specific wrapped module."""
        if not hasattr(self, '_wrapped_info'):
            raise ValueError("Model not wrapped yet. Call quantize() first.")
            
        if module_name not in self._wrapped_info:
            raise ValueError(f"Module {module_name} not found in wrapped modules")
        
        from ..wrapper import train_individual_wrapper
        
        wrapper_info = self._wrapped_info[module_name]
        
        # Load cached inputs
        block_inputs = CacheManager.load_block_cache(
            self._cached_inputs, module_name, cache_to_disk=self.cache_to_disk
        )
        
        if not block_inputs:
            print(f"Warning: No cached inputs found for {module_name}")
            return
        
        iters = additional_iters if additional_iters is not None else self.iters
        
        # Train the module
        result = train_individual_wrapper(
            module_name=module_name,
            wrapped_module=wrapper_info['wrapped_module'],
            block_inputs=block_inputs,
            iters=iters,
            lr=self.lr,
            batch_size=self.batch_size,
            is_single_layer=wrapper_info['is_single_layer']
        )
        
        if wrapper_info['is_single_layer'] and result is not None:
            _set_module(self.model, module_name, result)
            print(f"Applied trained quantization to {module_name}")
        else:
            from ..wrapper import WrapperLinear
            wrappers = []
            wrapped_module = wrapper_info['wrapped_module']
            for name, submodule in wrapped_module.named_modules():
                if isinstance(submodule, WrapperLinear):
                    wrappers.append(submodule)
            
            for wrapper in wrappers:
                wrapper.apply_best_params()
        
        # Auto-evaluate after training
        if auto_eval and self.tokenizer is not None:
            self.quick_eval(eval_samples=eval_samples)
    
    def train_all(self, auto_eval=True, eval_samples=5):
        """Train all wrapped modules."""
        if not hasattr(self, '_wrapped_info'):
            raise ValueError("Model not wrapped yet. Call quantize() first.")
        
        for i, module_name in enumerate(self._wrapped_info.keys()):
            self.train_module(module_name, auto_eval=auto_eval, eval_samples=eval_samples)
        
        if auto_eval and self.tokenizer is not None:
            self.quick_eval(eval_samples=eval_samples)
    
    def finalize(self):
        """Convert wrapped model to final QuantizedLinear model."""
        if not hasattr(self, '_wrapped_info'):
            raise ValueError("Model not wrapped yet. Call quantize() first.")
        
        from ..wrapper import finalize_wrapped_model
        
        try:
            quantized_model = finalize_wrapped_model(self.model, self._wrapped_info)
            
            # Clean up cache
            CacheManager.cleanup_all_cache(self._cached_inputs, cache_to_disk=self.cache_to_disk)
            
            # Clean up wrapper info
            delattr(self, '_wrapped_info')
            delattr(self, '_cached_inputs')
            
            print("✅ Model finalized and ready for inference!")
            return quantized_model
            
        finally:
            # Always cleanup
            if hasattr(self, '_cached_inputs'):
                CacheManager.cleanup_all_cache(self._cached_inputs, cache_to_disk=self.cache_to_disk)
    
    
    
    def _get_module(self, model, module_name: str):
        """Get a module by its dotted name."""
        return CacheManager._get_module(model, module_name)

