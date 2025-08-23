import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import QuantizedLinear
from .wrapper import WrapperLinear, wrapper_block, unwrapper_block
from .calib_dataset import get_calibration_dataset
from .utils.sign_sgd import SignSGD, CombinedScheduler
from .evaluation import calculate_perplexity, calculate_kl_divergence, print_report
import copy
import gc
from tqdm import tqdm
from typing import Dict, List, Optional

def _set_module(model, submodule_key, module):
    """
    Helper function to replace a module within a model hierarchy.
    
    Args:
        model (nn.Module): The main model.
        submodule_key (str): The dot-separated key to the submodule (e.g., "layer1.0.conv1").
        module (nn.Module): The new module to replace the old one.
    """
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)

class Bitween:
    """
    A simple quantizer class that takes a model and quantization configuration.
    """
    def __init__(self, model, tokenizer=None, bits=8, group_size=32, iters=1, lr=0.005, enable_minmax_tuning=True, seqlen=2048):
        self.model = model
        self.tokenizer = tokenizer
        self.bits = bits
        self.group_size = group_size
        
        # Training parameters for trainable quantization
        self.iters = iters  # Number of epochs (full passes through all samples) per block
        self.lr = lr        # Learning rate for optimization
        self.enable_minmax_tuning = enable_minmax_tuning  # Enable min/max scale tuning
        self.seqlen = seqlen  # Sequence length for calibration data

        assert group_size > 0, "Group size must be greater than 0."

        print(f"Bitween initialized with bits={bits}, group_size={group_size}")

    def quantize(self, evaluate_perplexity=False, num_samples=100, rtn=False, trainable=False, calib_dataset="pile-10k", nsamples=None, **eval_kwargs):
        """
        Quantizes the linear layers of the model.

        Args:
            evaluate_perplexity (bool): If True, run a full performance evaluation.
            num_samples (int): Number of samples to use for evaluation.
            rtn (bool): If True, use RTN quantization (fast, lower quality).
            trainable (bool): If True, use trainable quantization (slower, higher quality).
            calib_dataset (str): Dataset name for calibration ('pile-10k', etc.).
            nsamples (int): Number of calibration samples to use for training. If None, uses all available samples.
            **eval_kwargs: Additional arguments for the evaluation functions.
        
        Returns:
            Quantized model using QuantizedLinear layers with optimized parameters.
        """
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
                    print(f"  - Quantizing layer: {name}")
                    
                    q_module = QuantizedLinear.from_float(
                        module, self.bits, self.group_size
                    )
                    
                    _set_module(quantized_model, name, q_module)

        if trainable and not rtn:
            print(f"Trainable mode: iters={self.iters}, lr={self.lr}, minmax_tuning={self.enable_minmax_tuning}")
            # Trainable quantization: Use calibration data to optimize quantization parameters
            print(f"Using trainable quantization with {calib_dataset} dataset")
            quantized_model = self._trainable_quantize(calib_dataset, nsamples)
                
        print("--- Quantization complete ---")

        if evaluate_perplexity and self.tokenizer is not None and num_samples > 0:
            print("\n--- Evaluating quantized model ---")
            quantized_ppl = calculate_perplexity(quantized_model, self.tokenizer, num_samples=num_samples, **eval_kwargs)
            
            print("\n--- Calculating KL-Divergence ---")
            kl_div = calculate_kl_divergence(self.model, quantized_model, self.tokenizer, num_samples=num_samples, **eval_kwargs)
            
            print_report(original_ppl, quantized_ppl, kl_div)

        return quantized_model

    def _trainable_quantize(self, calib_dataset: str, nsamples: Optional[int]):
        """
        Performs trainable quantization using calibration dataset.
        
        This method implements the core trainable quantization workflow:
        1. Load calibration data and cache block inputs
        2. For each block: wrap → optimize → unwrap with best parameters
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
        print(f"Using {actual_nsamples} calibration samples for training")
        
        # Step 3: Cache intermediate data (inputs to each block) - ONCE for all blocks
        cached_inputs = self._cache_inter_data(quantized_model, calib_data, block_names, actual_nsamples)
        
        # Step 4: Quantize each block using cached inputs
        for block_name in block_names:
            print(f"\n--- Quantizing block: {block_name} ---")
            block = self._get_module(quantized_model, block_name)
            block_inputs = cached_inputs[block_name]
            self._quantize_block(block, block_inputs)
        
        return quantized_model
    
    def _get_block_names(self, model) -> List[str]:
        """
        Detect transformer blocks in the model architecture.
        
        This function identifies the main transformer blocks that should be
        quantized together. Different architectures have different naming patterns.
        
        Returns:
            List of block names (e.g., ['model.layers.0', 'model.layers.1', ...])
        """
        block_names = []
        
        # Common patterns for transformer blocks
        patterns = [
            'layers',      # LLaMA, Mistral: model.layers.0, model.layers.1, ...
            'h',           # GPT: transformer.h.0, transformer.h.1, ...
            'blocks',      # Some models: model.blocks.0, model.blocks.1, ...
            'decoder',     # T5: decoder.block.0, decoder.block.1, ...
        ]
        
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
                            break
                    break
        
        # Sort block names to ensure consistent processing order
        block_names.sort()
        
        # Fallback: if no blocks detected, quantize layer by layer
        if not block_names:
            print("Warning: No transformer blocks detected. Using layer-by-layer quantization.")
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    block_names.append(name)
        
        return block_names
    
    def _cache_inter_data(self, model, calib_data, block_names: List[str], nsamples: int) -> Dict[str, List[torch.Tensor]]:
        """
        Cache intermediate activations (inputs to each block) using calibration data.
        
        This is crucial for trainable quantization because we need to compare
        the outputs of original vs quantized blocks during optimization.
        
        Args:
            model: The model to analyze
            calib_data: Calibration dataset
            block_names: List of block names to cache inputs for
            
        Returns:
            Dictionary mapping block names to lists of input tensors
        """
        print("Caching intermediate activations...")
        
        cached_inputs = {name: [] for name in block_names}
        hooks = []
        
        def make_hook(block_name):
            def hook_fn(module, input, output):
                # Store the input to this block
                if isinstance(input, tuple):
                    cached_inputs[block_name].append(input[0].detach().cpu())
                else:
                    cached_inputs[block_name].append(input.detach().cpu())
            return hook_fn
        
        # Register hooks for each block
        for block_name in block_names:
            block = self._get_module(model, block_name)
            hook = block.register_forward_hook(make_hook(block_name))
            hooks.append(hook)
        
        # Run calibration data through model to collect inputs
        model.eval()
        with torch.no_grad():
            dataloader = calib_data.get_dataloader(batch_size=1)
            for i, batch in enumerate(tqdm(dataloader, desc="Caching block inputs", total=nsamples)):
                if i >= nsamples:
                    break
                    
                input_ids = batch['input_ids'].to(model.device)
                
                # Forward pass to trigger hooks and cache inputs for ALL blocks
                try:
                    _ = model(input_ids)
                except Exception as e:
                    print(f"Warning: Forward pass failed for batch {i}: {e}")
                    continue
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        print(f"Cached inputs for {len(block_names)} blocks")
        return cached_inputs
    
    def _quantize_block(self, block, block_inputs: List[torch.Tensor], batch_size: int = 32):
        """
        Memory-efficient quantization of a single block.
        
        Efficient approach:
        1. Cache original outputs ONCE using all available samples
        2. Wrap layers with learnable parameters (only for this block)
        3. Optimize using ALL cached samples for many iterations
        4. Unwrap and clean up before moving to next block
        
        Args:
            block: The transformer block to quantize
            block_inputs: List of ALL input tensors cached for this block
            batch_size: Mini-batch size for processing samples
        """
        if not block_inputs:
            print("Warning: No cached inputs for this block, skipping...")
            return
        
        num_samples = len(block_inputs)
        print(f"Optimizing block with {num_samples} cached samples")
        
        # Step 1: Cache original outputs using ALL samples
        print("Caching original block outputs for all samples...")
        original_outputs = []
        block.eval()
        
        with torch.no_grad():
            for inp in tqdm(block_inputs, desc="Caching original outputs", leave=False):
                inp = inp.to(next(block.parameters()).device)
                try:
                    orig_out = block(inp)
                    original_outputs.append(orig_out.detach().cpu())  # Store on CPU to save GPU memory
                except Exception as e:
                    print(f"Warning: Failed to cache output: {e}")
                    continue
        
        print(f"Cached {len(original_outputs)} original outputs")
        
        # Step 2: Wrap linear layers with learnable parameters (ONLY this block)
        quantized_names, unquantized_names = wrapper_block(
            block, 
            enable_minmax_tuning=self.enable_minmax_tuning,
            enable_round_tuning=True,
            bits=self.bits,
            group_size=self.group_size,
            device=next(block.parameters()).device
        )
        
        print(f"Wrapped {len(quantized_names)} linear layers")
        
        if not quantized_names:
            print("No linear layers found in this block")
            return
        
        # Step 3: Setup optimizer (ONLY for this block's parameters)
        learnable_params = []
        wrappers = []
        for name, module in block.named_modules():
            if isinstance(module, WrapperLinear):
                wrappers.append(module)
                if module.value is not None:
                    learnable_params.append(module.value)
                if module.min_scale is not None:
                    learnable_params.append(module.min_scale)
                if module.max_scale is not None:
                    learnable_params.append(module.max_scale)
        
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
                    inp = block_inputs[i].to(next(block.parameters()).device)
                    orig_out = original_outputs[i].to(inp.device)
                    
                    # Forward pass through quantized block
                    quant_out = block(inp)
                    
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
        for wrapper in wrappers:
            wrapper.apply_best_params()
        
        unwrapper_block(block, apply_quantization=True)
        print(f"Converted {len(quantized_names)} layers to QuantizedLinear")
        
        # Step 6: Clean up memory
        del original_outputs, learnable_params, wrappers, optimizer, scheduler
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def _get_module(self, model, module_name: str):
        """Get a module by its dotted name."""
        tokens = module_name.split('.')
        cur_mod = model
        for token in tokens:
            cur_mod = getattr(cur_mod, token)
        return cur_mod

