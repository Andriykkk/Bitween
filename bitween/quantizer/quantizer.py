import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules import QuantizedLinear
from ..wrapper import WrapperLinear, wrapper_block, unwrapper_block
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
                 cache_to_disk=False, max_memory_mb=512, ignore_layers=None, batch_size=32, chunk_size=10):
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
        self.chunk_size = chunk_size  # Number of samples to process at once during training
        
        # Layers to ignore during quantization
        self.ignore_layers = set(ignore_layers) if ignore_layers else set()
        self.batch_size = batch_size

        assert group_size > 0, "Group size must be greater than 0."

    def quantize(self, evaluate_perplexity=False, num_samples=100, rtn=False, trainable=False, calib_dataset="pile-10k", nsamples=None, 
                 cache_to_disk=None, max_memory_mb=None, ignore_layers=None, batch_size=None, chunk_size=None, **eval_kwargs):
        """
        Quantizes the linear layers of the model.

        Args:
            evaluate_perplexity (bool): If True, run a full performance evaluation.
            num_samples (int): Number of samples to use for evaluation.
            rtn (bool): If True, use RTN quantization (fast, lower quality).
            trainable (bool): If True, use trainable quantization (slower, higher quality).
            calib_dataset (str): Dataset name for calibration ('pile-10k', etc.).
            nsamples (int): Number of calibration samples to use for training. If None, uses all available samples.
            cache_to_disk (bool): Override instance setting for disk caching. If None, uses instance setting.
            max_memory_mb (int): Override instance setting for memory limit. If None, uses instance setting.
            ignore_layers (list): List of layer names to skip during quantization (e.g., ['lm_head', 'embed_tokens']).
            batch_size (int): Override batch size for training mini-batches. If None, uses instance setting.
            chunk_size (int): Override chunk size for chunked processing. If None, uses instance setting.
            **eval_kwargs: Additional arguments for the evaluation functions.
        
        Returns:
            Quantized model using QuantizedLinear layers with optimized parameters.
        """
        # Handle override parameters - temporarily update instance variables
        original_ignore_layers = self.ignore_layers
        original_chunk_size = self.chunk_size
        
        if ignore_layers is not None:
            self.ignore_layers = set(ignore_layers)
        if batch_size is not None:
            self.batch_size = batch_size
        if chunk_size is not None:
            self.chunk_size = chunk_size
            
        original_ppl = None
        if evaluate_perplexity and self.tokenizer is not None and num_samples > 0:
            if self.tokenizer is None:
                raise ValueError("Tokenizer must be provided for evaluation.")
            print("\n--- Evaluating original model ---")
            original_ppl = calculate_perplexity(self.model, self.tokenizer, num_samples=num_samples, **eval_kwargs)

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
                
            print(f"Trainable mode: iters={self.iters}, lr={self.lr}, minmax_tuning={self.enable_minmax_tuning}")
            print(f"Cache mode: {'disk' if self.cache_to_disk else 'memory'} (max: {self.max_memory_mb}MB)")
            # Trainable quantization: Use calibration data to optimize quantization parameters
            print(f"Using trainable quantization with {calib_dataset} dataset")
            quantized_model = self._trainable_quantize(calib_dataset, nsamples, self.batch_size)
                
        print("--- Quantization complete ---")

        if evaluate_perplexity and self.tokenizer is not None and num_samples > 0:
            print("\n--- Evaluating quantized model ---")
            quantized_ppl = calculate_perplexity(quantized_model, self.tokenizer, num_samples=num_samples, **eval_kwargs)
            
            print("\n--- Calculating KL-Divergence ---")
            kl_div, token_kl_div = calculate_kl_divergence(self.model, quantized_model, self.tokenizer, num_samples=num_samples, **eval_kwargs)
            
            print_report(original_ppl, quantized_ppl, kl_div, token_kl_div)

        # Restore original settings
        self.ignore_layers = original_ignore_layers
        self.chunk_size = original_chunk_size
        return quantized_model

    def _trainable_quantize(self, calib_dataset: str, nsamples: Optional[int], batch_size: int):
        """
        Performs trainable quantization using calibration dataset with chunked processing.
        
        This method implements chunk-based quantization workflow:
        1. Load calibration data
        2. For each block: process samples in chunks, train incrementally
        3. Return model with optimized QuantizedLinear layers
        
        Args:
            calib_dataset (str): Name of calibration dataset
            nsamples (Optional[int]): Number of calibration samples. If None, uses all available.
            
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
        # If nsamples is None, load all available samples from dataset
        dataset_nsamples = nsamples if nsamples is not None else 10000  # Default pile-10k size
        calib_data = get_calibration_dataset(
            dataset_name=calib_dataset,
            tokenizer=self.tokenizer,
            seqlen=self.seqlen,
            nsamples=dataset_nsamples
        )
        
        # Use all loaded samples for training
        actual_nsamples = len(calib_data)
        
        # Validate and adjust chunk_size
        effective_chunk_size = self._validate_chunk_size(actual_nsamples, batch_size)
        use_chunking = effective_chunk_size is not None
        
        if use_chunking:
            print(f"Using {actual_nsamples} calibration samples for chunked training (chunk_size={effective_chunk_size})")
        else:
            print(f"Using {actual_nsamples} calibration samples for full caching training")
        
        # Step 3: Quantize each block/layer 
        if use_chunking:
            # Use chunked processing
            for block_name in block_names:
                print(f"\n--- Quantizing: {block_name} ---")
                block_or_layer = self._get_module(quantized_model, block_name)
                
                result = self._quantize_block_chunked(block_or_layer, calib_data, block_name, actual_nsamples, batch_size, effective_chunk_size)
                
                if isinstance(block_or_layer, nn.Linear) and result is not None:
                    _set_module(quantized_model, block_name, result)
                    print(f"Replaced {block_name} with quantized version")
        else:
            # Use original full caching method
            cached_inputs = CacheManager.cache_block_inputs(
                quantized_model, calib_data, block_names, actual_nsamples,
                cache_to_disk=self.cache_to_disk, max_memory_mb=self.max_memory_mb
            )
            
            try:
                for block_name in block_names:
                    print(f"\n--- Quantizing: {block_name} ---")
                    block_or_layer = self._get_module(quantized_model, block_name)
                    
                    # Load cached inputs for this block/layer
                    block_inputs = CacheManager.load_block_cache(cached_inputs, block_name, cache_to_disk=self.cache_to_disk)
                    
                    if not block_inputs:
                        print(f"Warning: No cached inputs found for {block_name}, skipping...")
                        continue
                    
                    # Quantize the block or individual layer
                    result = self._quantize_block(block_or_layer, block_inputs, block_name, batch_size)
                    
                    # If it's a single layer, we need to replace it in the model
                    if isinstance(block_or_layer, nn.Linear) and result is not None:
                        _set_module(quantized_model, block_name, result)
                        print(f"Replaced {block_name} with quantized version")
                    
                    # Clean up cache for this block to free memory
                    CacheManager.cleanup_block_cache(cached_inputs, block_name, cache_to_disk=self.cache_to_disk)
            
            finally:
                # Always clean up all cache files when done
                CacheManager.cleanup_all_cache(cached_inputs, cache_to_disk=self.cache_to_disk)
        
        return quantized_model
    
    def _validate_chunk_size(self, nsamples: int, batch_size: int) -> Optional[int]:
        """
        Validate and adjust chunk_size based on constraints.
        
        Args:
            nsamples: Total number of samples
            batch_size: Training batch size
            
        Returns:
            Valid chunk_size or None if chunking should be disabled
        """
        # If chunk_size is None or 0, disable chunking
        if self.chunk_size is None or self.chunk_size <= 0:
            return None
        
        # If chunk_size >= nsamples, disable chunking (process all at once)
        if self.chunk_size >= nsamples:
            return None
        
        # Ensure chunk_size is at least as large as batch_size
        effective_chunk_size = max(self.chunk_size, batch_size)
        
        # Make chunk_size divisible by batch_size
        if effective_chunk_size % batch_size != 0:
            effective_chunk_size = ((effective_chunk_size + batch_size - 1) // batch_size) * batch_size
        
        # Final check: if adjusted chunk_size >= nsamples, disable chunking
        if effective_chunk_size >= nsamples:
            return None
            
        return effective_chunk_size
    
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
    
    def _quantize_block_chunked(self, block_or_layer, calib_data, block_name: str, total_nsamples: int, batch_size: int, chunk_size: int):
        """
        Memory-efficient chunked quantization of a single block or individual layer.
        
        Processes samples in small chunks, caching only chunk_size samples at a time:
        1. Run model on chunk_size samples to get inputs/outputs for this block
        2. Train on this chunk
        3. Repeat for next chunk
        
        Args:
            block_or_layer: Either a transformer block module or a single linear layer
            calib_data: Calibration dataset 
            block_name: Name of the block/layer for logging
            total_nsamples: Total number of samples to process
            batch_size: Training batch size
        """
        # Detect if this is a single linear layer or a block
        is_single_layer = isinstance(block_or_layer, nn.Linear)
        item_type = "linear layer" if is_single_layer else "block"
        
        # Step 1: Setup wrappers
        if is_single_layer:
            # Check if this single layer should be ignored
            if block_name in self.ignore_layers:
                print(f"Skipping ignored standalone layer: {block_name}")
                return None
                
            # For individual linear layers, create a temporary wrapper
            wrapper = WrapperLinear(
                block_or_layer,
                bits=self.bits,
                group_size=self.group_size,
                enable_minmax_tuning=self.enable_minmax_tuning,
                enable_round_tuning=True,
                device=next(block_or_layer.parameters()).device
            )
            quantized_names = ["single_layer"]
            wrappers = [wrapper]
        else:
            # For blocks, wrap all linear layers inside
            quantized_names, unquantized_names = wrapper_block(
                block_or_layer, 
                enable_minmax_tuning=self.enable_minmax_tuning,
                enable_round_tuning=True,
                bits=self.bits,
                group_size=self.group_size,
                device=next(block_or_layer.parameters()).device,
                ignore_layers=self.ignore_layers,
                block_prefix=block_name
            )
            
            # Collect wrappers
            wrappers = []
            for name, module in block_or_layer.named_modules():
                if isinstance(module, WrapperLinear):
                    wrappers.append(module)
        
        if not quantized_names:
            print(f"No linear layers found in this {item_type}")
            return None
        
        # Step 2: Setup optimizer parameters
        learnable_params = []
        for wrapper in wrappers:
            if wrapper.value is not None:
                learnable_params.append(wrapper.value)
            if wrapper.min_scale is not None:
                learnable_params.append(wrapper.min_scale)
            if wrapper.max_scale is not None:
                learnable_params.append(wrapper.max_scale)
        
        if not learnable_params:
            print("Warning: No learnable parameters found")
            return None
        
        # Setup SignSGD optimizer
        optimizer = SignSGD(learnable_params, lr=self.lr, momentum=0.9)
        
        # Calculate training parameters
        num_chunks = (total_nsamples + chunk_size - 1) // chunk_size
        total_steps = self.iters * num_chunks
        warmup_steps = total_steps // 10
        
        scheduler = CombinedScheduler(
            optimizer, 
            target_lr=self.lr,
            warmup_steps=warmup_steps,
            patience=total_steps // 20,
            factor=0.8,
            min_lr=self.lr * 0.01,
            verbose=True
        )
        
        print(f"Chunked training: {self.iters} epochs × {num_chunks} chunks = {total_steps} total steps")
        print(f"Chunk size: {chunk_size}, Total samples: {total_nsamples}")
        
        # Step 3: Training loop with chunked processing
        best_loss = float('inf')
        dataloader = calib_data.get_dataloader(batch_size=1)
        global_step = 0
        
        for epoch in range(self.iters):
            epoch_loss = 0.0
            chunk_count = 0
            
            # Process data in chunks
            data_iterator = iter(dataloader)
            
            for chunk_start in range(0, total_nsamples, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_nsamples)
                actual_chunk_size = chunk_end - chunk_start
                
                print(f"Processing chunk {chunk_count + 1}/{num_chunks} (samples {chunk_start}-{chunk_end-1})")
                
                # Step 3a: Cache inputs and outputs for this chunk
                chunk_inputs = []
                chunk_outputs = []
                
                # Register hook to capture inputs
                inputs_captured = []
                def capture_hook(module, input, output):
                    if isinstance(input, tuple):
                        tensor = input[0].detach().cpu()
                    else:
                        tensor = input.detach().cpu()
                    inputs_captured.append(tensor)
                
                hook = block_or_layer.register_forward_hook(capture_hook)
                
                # Process chunk samples to get inputs/outputs
                block_or_layer.eval()
                with torch.no_grad():
                    for _ in range(actual_chunk_size):
                        try:
                            batch = next(data_iterator)
                            input_ids = batch['input_ids'].to(self.model.device)
                            
                            # Forward pass to get input to this block
                            _ = self.model(input_ids)
                            
                            # Get the captured input and original output
                            if inputs_captured:
                                inp = inputs_captured[-1].to(next(block_or_layer.parameters()).device)
                                chunk_inputs.append(inp)
                                
                                # Get original output
                                orig_out = block_or_layer(inp)
                                if isinstance(orig_out, tuple):
                                    orig_out = orig_out[0]
                                chunk_outputs.append(orig_out.detach().cpu())
                                
                        except StopIteration:
                            break
                        except Exception as e:
                            print(f"Warning: Failed to process sample: {e}")
                            continue
                
                hook.remove()
                
                # Step 3b: Train on this chunk
                if chunk_inputs and chunk_outputs:
                    chunk_loss = self._train_on_chunk(
                        block_or_layer, chunk_inputs, chunk_outputs, 
                        optimizer, scheduler, is_single_layer, wrapper if is_single_layer else None,
                        wrappers, batch_size, global_step
                    )
                    
                    epoch_loss += chunk_loss * len(chunk_inputs)
                    global_step += 1
                    
                    if chunk_loss < best_loss:
                        best_loss = chunk_loss
                        for w in wrappers:
                            w.update_best_params(chunk_loss)
                
                # Clean up chunk memory
                del chunk_inputs, chunk_outputs, inputs_captured
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                chunk_count += 1
            
            # Epoch statistics
            avg_epoch_loss = epoch_loss / total_nsamples if total_nsamples > 0 else 0
            current_lr = scheduler.get_lr()[0]
            print(f"Epoch {epoch + 1}/{self.iters}: Avg Loss={avg_epoch_loss:.6f}, LR={current_lr:.6f}")
        
        print(f"Chunked optimization complete. Best loss: {best_loss:.6f}")
        
        # Step 4: Apply best parameters and convert to QuantizedLinear
        for wrapper_inst in wrappers:
            wrapper_inst.apply_best_params()
        
        if is_single_layer:
            quantized_layer = wrapper.to_quantized_linear()
            del wrapper
            return quantized_layer
        else:
            unwrapper_block(block_or_layer, apply_quantization=True)
            return None
    
    def _train_on_chunk(self, block_or_layer, chunk_inputs, chunk_outputs, optimizer, scheduler, 
                       is_single_layer, wrapper, wrappers, batch_size, global_step):
        """Train on a single chunk of data."""
        optimizer.zero_grad()
        total_loss = 0.0
        num_samples = len(chunk_inputs)
        
        # Process mini-batches within this chunk
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_loss = 0.0
            
            for i in range(batch_start, batch_end):
                inp = chunk_inputs[i].to(next(block_or_layer.parameters()).device)
                orig_out = chunk_outputs[i].to(inp.device)
                
                # Forward pass through quantized block/layer
                if is_single_layer:
                    quant_out = wrapper(inp)
                else:
                    quant_out = block_or_layer(inp)
                
                # Handle tuple outputs
                if isinstance(quant_out, tuple):
                    quant_out = quant_out[0]
                
                # MSE loss
                loss = F.mse_loss(quant_out, orig_out)
                batch_loss += loss.item()
                loss.backward()
            
            total_loss += batch_loss
        
        # Update parameters
        optimizer.step()
        avg_loss = total_loss / num_samples
        scheduler.step(avg_loss, global_step)
        
        return avg_loss
    
    
    def _quantize_block(self, block_or_layer, block_inputs: List[torch.Tensor], block_name: str, batch_size: int):
        """
        Memory-efficient quantization of a single block or individual layer.
        
        This method handles both:
        1. Transformer blocks (containing multiple linear layers)
        2. Individual linear layers (standalone layers not in blocks)
        
        Args:
            block_or_layer: Either a transformer block module or a single linear layer
            block_inputs: List of ALL input tensors cached for this block/layer
            block_name: Name of the block/layer for logging and ignore checking
        """
        if not block_inputs:
            print("Warning: No cached inputs for this block, skipping...")
            return
        
        num_samples = len(block_inputs)
        
        # Detect if this is a single linear layer or a block
        is_single_layer = isinstance(block_or_layer, nn.Linear)
        item_type = "linear layer" if is_single_layer else "block"
        
        original_outputs = []
        block_or_layer.eval()
        
        with torch.no_grad():
            for inp in tqdm(block_inputs, desc="Caching original outputs", leave=False):
                inp = inp.to(next(block_or_layer.parameters()).device)
                try:
                    orig_out = block_or_layer(inp)
                    
                    # Handle different output formats (tensor vs tuple)
                    if isinstance(orig_out, tuple):
                        # Take the first element (usually the main output tensor)
                        orig_out = orig_out[0]
                    
                    original_outputs.append(orig_out.detach().cpu())  # Store on CPU to save GPU memory
                except Exception as e:
                    print(f"Warning: Failed to cache output: {e}")
                    continue
        
        # Step 2: Handle wrapping based on type
        if is_single_layer:
            # Check if this single layer should be ignored
            if block_name in self.ignore_layers:
                print(f"Skipping ignored standalone layer: {block_name}")
                return  # Skip this layer entirely
                
            # For individual linear layers, create a temporary wrapper
            wrapper = WrapperLinear(
                block_or_layer,
                bits=self.bits,
                group_size=self.group_size,
                enable_minmax_tuning=self.enable_minmax_tuning,
                enable_round_tuning=True,
                device=next(block_or_layer.parameters()).device
            )
            quantized_names = ["single_layer"]
            wrappers = [wrapper]
        else:
            # For blocks, wrap all linear layers inside
            quantized_names, unquantized_names = wrapper_block(
                block_or_layer, 
                enable_minmax_tuning=self.enable_minmax_tuning,
                enable_round_tuning=True,
                bits=self.bits,
                group_size=self.group_size,
                device=next(block_or_layer.parameters()).device,
                ignore_layers=self.ignore_layers,
                block_prefix=block_name
            )
            
            # Collect wrappers
            wrappers = []
            for name, module in block_or_layer.named_modules():
                if isinstance(module, WrapperLinear):
                    wrappers.append(module)
        
        if not quantized_names:
            print(f"No linear layers found in this {item_type}")
            return
        
        # Step 3: Setup optimizer parameters
        learnable_params = []
        for wrapper in wrappers:
            if wrapper.value is not None:
                learnable_params.append(wrapper.value)
            if wrapper.min_scale is not None:
                learnable_params.append(wrapper.min_scale)
            if wrapper.max_scale is not None:
                learnable_params.append(wrapper.max_scale)
        
        if not learnable_params:
            print("Warning: No learnable parameters found")
            return
        
        # Setup SignSGD optimizer with adaptive learning rate scheduling
        optimizer = SignSGD(learnable_params, lr=self.lr, momentum=0.9)
        
        # Calculate total optimizer steps: epochs × samples_per_epoch
        # Each epoch processes all samples in mini-batches, then updates parameters
        batch_size = min(batch_size, num_samples)  # Process in mini-batches to avoid memory issues
        steps_per_epoch = (num_samples + batch_size - 1) // batch_size
        total_steps = self.iters * steps_per_epoch
        
        # Setup combined scheduler based on actual optimizer steps
        warmup_steps = total_steps // 10  # Warmup for 10% of steps
        scheduler = CombinedScheduler(
            optimizer, 
            target_lr=self.lr,
            warmup_steps=warmup_steps,
            patience=total_steps // 20,  # Patience for 5% of steps  
            factor=0.8,
            min_lr=self.lr * 0.01,
            verbose=True
        )
        
        print(f"Training setup: {self.iters} epochs × {steps_per_epoch} steps/epoch = {total_steps} total steps")
        print(f"Batch size: {batch_size}, Warmup: {warmup_steps} steps, Patience: {max(steps_per_epoch, total_steps // 20)} steps")
        
        # Step 4: Training loop - Process samples in mini-batches
        best_loss = float('inf')
        loss_history = []
        global_step = 0
        
        for epoch in range(self.iters):
            epoch_loss = 0.0
            
            # Process all samples in mini-batches for this epoch
            for batch_start in range(0, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)
                
                # Clear gradients once per mini-batch
                optimizer.zero_grad()
                batch_loss = 0.0
                
                # Process mini-batch and accumulate gradients
                for i in range(batch_start, batch_end):
                    inp = block_inputs[i].to(next(block_or_layer.parameters()).device)
                    orig_out = original_outputs[i].to(inp.device)
                    
                    # Forward pass through quantized block/layer
                    if is_single_layer:
                        # For single layer, use the wrapper directly
                        quant_out = wrapper(inp)
                    else:
                        # For blocks, use the block with wrapped layers
                        quant_out = block_or_layer(inp)
                    
                    # Handle tuple outputs (extract main tensor)
                    if isinstance(quant_out, tuple):
                        quant_out = quant_out[0]
                    
                    # MSE loss between original and quantized outputs
                    loss = F.mse_loss(quant_out, orig_out)
                    batch_loss += loss.item()
                    
                    # Backward pass - accumulate gradients for this mini-batch
                    loss.backward()
                
                # Update parameters using SignSGD with accumulated gradients
                optimizer.step()
                global_step += 1
                
                # Update learning rate scheduler
                avg_batch_loss = batch_loss / (batch_end - batch_start)
                scheduler.step(avg_batch_loss, global_step)
                epoch_loss += batch_loss
                
                # Track best parameters
                if avg_batch_loss < best_loss:
                    best_loss = avg_batch_loss
                    for wrapper in wrappers:
                        wrapper.update_best_params(avg_batch_loss)
            
            # Epoch statistics
            avg_epoch_loss = epoch_loss / num_samples
            loss_history.append(avg_epoch_loss)
            
            # Progress reporting
            current_lr = scheduler.get_lr()[0]
            print(f"Epoch {epoch + 1}/{self.iters}: Avg Loss={avg_epoch_loss:.6f}, LR={current_lr:.6f}, Steps={global_step}")
        
        # Final statistics
        improvement = (loss_history[0] - best_loss) / loss_history[0] * 100 if loss_history else 0
        print(f"Block optimization complete. Best loss: {best_loss:.6f} ({improvement:.1f}% improvement)")
        
        # Step 5: Apply best parameters and convert to QuantizedLinear
        for wrapper_inst in wrappers:
            wrapper_inst.apply_best_params()
        
        if is_single_layer:
            quantized_layer = wrapper.to_quantized_linear()
        else:
            unwrapper_block(block_or_layer, apply_quantization=True)
        
        # Step 6: Clean up memory aggressively
        del original_outputs, learnable_params, wrappers, optimizer, scheduler
        if is_single_layer:
            del wrapper  # Clean up the wrapper reference
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Return quantized layer for single layer case
        if is_single_layer:
            return quantized_layer
    
    def _get_module(self, model, module_name: str):
        """Get a module by its dotted name."""
        return CacheManager._get_module(model, module_name)

