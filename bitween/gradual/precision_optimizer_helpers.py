"""
Helper functions for PrecisionOptimizer to handle frozen layers.
"""

def _get_frozen_layer_names_for_block(self, block_name):
    """Get list of frozen layer names for a block."""
    if block_name not in self.block_quantizations:
        return []
        
    config = self.block_quantizations[block_name]
    if not config.layer_exceptions:
        return []
        
    # Return layers that are marked as frozen
    frozen_layers = []
    for layer_name, exception_info in config.layer_exceptions.items():
        if isinstance(exception_info, dict) and exception_info.get('reason') == 'quality_recovery':
            frozen_layers.append(layer_name)
        elif exception_info == 'frozen':  # Legacy format
            frozen_layers.append(layer_name)
            
    return frozen_layers
    
def _train_wrapper_respecting_frozen_layers(self, module_name, wrapped_module, block_inputs, iters, lr, batch_size, is_single_layer):
    """Unified training function that automatically excludes frozen layers from optimizer."""
    from ..utils.sign_sgd import SignSGD
    from ..wrapper import WrapperLinear
    from tqdm import tqdm
    import torch
    
    # Get frozen layer names for this block
    frozen_layer_names = self._get_frozen_layer_names_for_block(module_name)
    
    if frozen_layer_names:
        print(f"        Training {module_name} with {len(frozen_layer_names)} frozen layers: {frozen_layer_names}")
    else:
        print(f"        Training {module_name} (no frozen layers)")
    
    if not block_inputs:
        print("Warning: No cached inputs for training, skipping...")
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
                orig_out = wrapped_module(inp)
                if isinstance(orig_out, tuple):
                    orig_out = orig_out[0]
                original_outputs.append(orig_out.detach().cpu())
            except Exception as e:
                print(f"Warning: Failed to cache output: {e}")
                continue
    
    if not original_outputs:
        print("Warning: No original outputs cached")
        return None
    
    # Find all wrapper instances
    wrappers = []
    if isinstance(wrapped_module, WrapperLinear):
        wrappers = [wrapped_module]
    else:
        for name, submodule in wrapped_module.named_modules():
            if isinstance(submodule, WrapperLinear):
                wrappers.append(submodule)
    
    # Setup optimizer parameters (EXCLUDE frozen layers)
    learnable_params = []
    trained_wrapper_count = 0
    
    for wrapper in wrappers:
        # Get wrapper's name in the module hierarchy
        wrapper_name = None
        for name, submodule in wrapped_module.named_modules():
            if submodule is wrapper:
                wrapper_name = name
                break
        
        # Skip if this wrapper is frozen
        if wrapper_name in frozen_layer_names:
            print(f"        Skipping frozen layer: {wrapper_name}")
            continue
        
        # Add trainable parameters for non-frozen layers
        if wrapper.value is not None:
            learnable_params.append(wrapper.value)
        if wrapper.min_scale is not None:
            learnable_params.append(wrapper.min_scale)
        if wrapper.max_scale is not None:
            learnable_params.append(wrapper.max_scale)
        
        trained_wrapper_count += 1
    
    if not learnable_params:
        print(f"        Warning: No learnable parameters found (all {len(wrappers)} layers frozen?)")
        return wrapped_module
    
    print(f"        Training {trained_wrapper_count}/{len(wrappers)} layers ({len(wrappers)-trained_wrapper_count} frozen)")
    
    # Create optimizer with only non-frozen parameters
    optimizer = SignSGD(learnable_params, lr=lr, momentum=0.9)
    
    batch_size = min(batch_size, num_samples)
    steps_per_epoch = (num_samples + batch_size - 1) // batch_size
    total_steps = iters * steps_per_epoch
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps
    )
    
    wrapped_module.train()
    
    # Training loop
    for epoch in range(iters):
        epoch_loss = 0.0
        indices = torch.randperm(num_samples)
        
        for step in range(steps_per_epoch):
            start_idx = step * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            batch_loss = 0.0
            
            for idx in batch_indices:
                inp = block_inputs[idx].to(device)
                target = original_outputs[idx].to(device)
                
                optimizer.zero_grad()
                
                output = wrapped_module(inp)
                if isinstance(output, tuple):
                    output = output[0]
                
                loss = torch.nn.functional.mse_loss(output, target)
                loss.backward()
                
                optimizer.step()
                scheduler.step()
                
                batch_loss += loss.item()
            
            epoch_loss += batch_loss / len(batch_indices)
        
        avg_loss = epoch_loss / steps_per_epoch
        print(f"        Epoch {epoch+1}/{iters}: Loss = {avg_loss:.6f}, LR = {scheduler.get_last_lr()[0]:.6f}")
    
    wrapped_module.eval()
    return wrapped_module
    
def _apply_best_params_respecting_frozen_layers(self, block_name, wrapped_module):
    """Apply best parameters only to non-frozen layers."""
    from ..wrapper import WrapperLinear
    
    frozen_layer_names = self._get_frozen_layer_names_for_block(block_name)
    
    for name, submodule in wrapped_module.named_modules():
        if isinstance(submodule, WrapperLinear):
            if name not in frozen_layer_names:
                submodule.apply_best_params()
            else:
                print(f"        Skipping parameter application for frozen layer: {name}")