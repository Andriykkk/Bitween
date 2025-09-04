import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules import QuantizedLinear
from ..wrapper import WrapperLinear, unified_wrapper, unified_unwrapper
from ..calib_dataset import get_calibration_dataset
from ..utils.sign_sgd import SignSGD, CombinedScheduler
from ..utils.evaluation import calculate_perplexity, calculate_kl_divergence, print_report
import copy
import gc
from tqdm import tqdm
from typing import Dict, List, Optional
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
        Memory-efficient quantization using unified wrapper approach.
        
        Args:
            module: Module to quantize (block or individual layer)
            block_inputs: List of ALL input tensors cached for this module
            module_name: Name of the module for logging and ignore checking
            batch_size: Training batch size
        """
        if not block_inputs:
            print("Warning: No cached inputs for this module, skipping...")
            return
        
        num_samples = len(block_inputs)
        
        # Cache original outputs
        original_outputs = []
        module.eval()
        
        with torch.no_grad():
            for inp in tqdm(block_inputs, desc="Caching original outputs", leave=False):
                inp = inp.to(next(module.parameters()).device)
                try:
                    orig_out = module(inp)
                    
                    # Handle different output formats (tensor vs tuple)
                    if isinstance(orig_out, tuple):
                        orig_out = orig_out[0]
                    
                    original_outputs.append(orig_out.detach().cpu())
                except Exception as e:
                    print(f"Warning: Failed to cache output: {e}")
                    continue
        
        # Step 2: Wrap using unified wrapper
        wrapped_module, quantized_names = unified_wrapper(
            module,
            enable_minmax_tuning=self.enable_minmax_tuning,
            enable_round_tuning=True,
            bits=self.bits,
            group_size=self.group_size,
            device=next(module.parameters()).device,
            ignore_layers=self.ignore_layers,
            module_prefix=module_name
        )
        
        if not quantized_names:
            print(f"No linear layers found to quantize in: {module_name}")
            return
        
        # Check perplexity after wrapping (before training) to verify wrapper effect
        print(f"Checking {module_name} after wrapping...")
        self._check_module_output_similarity(module, wrapped_module, block_inputs[:min(5, len(block_inputs))])
        
        # Collect wrappers
        wrappers = []
        if isinstance(wrapped_module, WrapperLinear):
            wrappers = [wrapped_module]
        else:
            for name, submodule in wrapped_module.named_modules():
                if isinstance(submodule, WrapperLinear):
                    wrappers.append(submodule)
        
        # Step 3: Setup optimizer parameters
        learnable_params = []
        param_info = []
        for i, wrapper in enumerate(wrappers):
            if wrapper.value is not None:
                learnable_params.append(wrapper.value)
                param_info.append(f"wrapper{i}_value: {wrapper.value.numel()} params")
            if wrapper.min_scale is not None:
                learnable_params.append(wrapper.min_scale)
                param_info.append(f"wrapper{i}_min_scale: {wrapper.min_scale.numel()} params")
            if wrapper.max_scale is not None:
                learnable_params.append(wrapper.max_scale)
                param_info.append(f"wrapper{i}_max_scale: {wrapper.max_scale.numel()} params")
        
        if not learnable_params:
            print("Warning: No learnable parameters found")
            return
        
        optimizer = SignSGD(learnable_params, lr=self.lr, momentum=0.9)
        
        batch_size = min(batch_size, num_samples)  # Process in mini-batches to avoid memory issues
        steps_per_epoch = (num_samples + batch_size - 1) // batch_size
        total_steps = self.iters * steps_per_epoch
        
        # Use LinearLR scheduler like AutoRound (1.0 -> 0.0 over total_steps)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps
        )
        
        print(f"Training setup: {self.iters} epochs × {steps_per_epoch} steps/epoch = {total_steps} total steps")
        # print(f"Batch size: {batch_size}, Warmup: {warmup_steps} steps, Patience: {max(steps_per_epoch, total_steps // 20)} steps")
        
        # Step 4: Training loop - Process samples in mini-batches
        best_loss = float('inf')
        loss_history = []
        global_step = 0
        step_losses = []  # Track loss at each step
        
        print(f"Starting training loop: {self.iters} epochs, {steps_per_epoch} steps/epoch")
        
        for epoch in range(self.iters):
            epoch_loss = 0.0
            epoch_step_losses = []
            
            # Process all samples in mini-batches for this epoch
            for batch_start in range(0, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)
                
                # Clear gradients once per mini-batch
                optimizer.zero_grad()
                batch_loss = 0.0
                
                # Process mini-batch and accumulate gradients
                for i in range(batch_start, batch_end):
                    inp = block_inputs[i].to(next(wrapped_module.parameters()).device)
                    orig_out = original_outputs[i].to(inp.device)
                    
                    # Forward pass through quantized module
                    quant_out = wrapped_module(inp)
                    
                    # Handle tuple outputs (extract main tensor)
                    if isinstance(quant_out, tuple):
                        quant_out = quant_out[0]
                    
                    # MSE loss between original and quantized outputs
                    loss = F.mse_loss(quant_out, orig_out)
                    
                    # AutoRound approach: Scale loss by 1000 for better gradients
                    scaled_loss = loss * 1000
                    batch_loss += scaled_loss.item()  # Store scaled loss for reporting
                    
                    # Backward pass - accumulate gradients for this mini-batch
                    scaled_loss.backward()
                
                # Check gradient magnitudes before optimizer step
                grad_info = []
                for i, param in enumerate(learnable_params):
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        grad_info.append(f"param{i}: {grad_norm:.6f}")
                    else:
                        grad_info.append(f"param{i}: NO_GRAD")
                
                # Update parameters using SignSGD with accumulated gradients
                optimizer.step()
                global_step += 1
                
                # Update learning rate scheduler (AutoRound style)
                scheduler.step()
                
                avg_batch_loss = batch_loss / (batch_end - batch_start)
                epoch_loss += batch_loss
                step_losses.append(avg_batch_loss)
                epoch_step_losses.append(avg_batch_loss)
                
                # Print step-by-step progress every few steps
                if global_step % max(1, steps_per_epoch // 4) == 0 or global_step <= 3:
                    current_lr = scheduler.get_lr()[0]
                    print(f"  Step {global_step}/{total_steps}: loss={avg_batch_loss:.6f}, lr={current_lr:.6f}")
            
            # Epoch statistics
            avg_epoch_loss = epoch_loss / num_samples
            loss_history.append(avg_epoch_loss)
            
            # Detailed epoch reporting
            epoch_min = min(epoch_step_losses) if epoch_step_losses else 0
            epoch_max = max(epoch_step_losses) if epoch_step_losses else 0
            current_lr = scheduler.get_lr()[0]
            
            print(f"Epoch {epoch + 1}/{self.iters}: Avg={avg_epoch_loss:.6f}, Min={epoch_min:.6f}, Max={epoch_max:.6f}, LR={current_lr:.6f}")
            
        best_loss = avg_batch_loss
        # Final statistics with more detail
        print(f"Training complete. Final loss: {best_loss:.6f}")
        
        # Step 5: Apply best parameters and convert to QuantizedLinear
        for wrapper_inst in wrappers:
            wrapper_inst.apply_best_params()
        
        # Unwrap using unified unwrapper
        result = unified_unwrapper(wrapped_module, apply_quantization=True)
        
        # Step 6: Clean up memory aggressively
        del original_outputs, learnable_params, wrappers, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Return quantized layer for single layer case, None for block case
        return result if isinstance(wrapped_module, WrapperLinear) else None
    
    def _check_module_output_similarity(self, orig_module, wrapped_module, sample_inputs):
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
    
    def _get_module(self, model, module_name: str):
        """Get a module by its dotted name."""
        return CacheManager._get_module(model, module_name)

