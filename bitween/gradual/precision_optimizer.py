"""
Handles precision optimization with group size co-optimization and rollback mechanisms.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum


class OptimizationResult(Enum):
    """Results of quantization optimization attempts."""
    SUCCESS = "success"
    FAILED_RTN = "failed_rtn"
    FAILED_TRAINING = "failed_training" 
    FAILED_RECOVERY = "failed_recovery"
    NO_IMPROVEMENT = "no_improvement"


@dataclass
class QuantizationConfig:
    """Configuration for a quantization attempt."""
    bits: int
    group_size: int
    method: str  # "RTN", "Trained", "Mixed"
    error_metric: float
    layer_exceptions: Dict = None  # Layers that couldn't be quantized or need higher precision
    training_metadata: Dict = None  # Training details, epochs, loss curves, etc.
    memory_savings: float = 0.0    # Estimated memory reduction
    
    def __post_init__(self):
        if self.layer_exceptions is None:
            self.layer_exceptions = {}
        if self.training_metadata is None:
            self.training_metadata = {}


class PrecisionOptimizer:
    """
    Handles the actual quantization process with precision and group size optimization.
    
    Coordinates with existing Bitween utilities for RTN and trainable quantization.
    Manages rollback when optimization fails to meet budget constraints.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        tokenizer, 
        min_group_size: int = 32, 
        max_group_size: int = 128,
        baseline_metrics: Dict = None,
        evaluation_samples: int = 5,
        budget_allocations: Dict = None,
        original_model: nn.Module = None,
        training_batch_size: int = 4,
        training_manager = None
    ):
        """
        Initialize precision optimizer.
        
        Args:
            model: The model to optimize
            tokenizer: Tokenizer for the model
            min_group_size: Minimum group size for quantization
            max_group_size: Maximum group size for quantization
            baseline_metrics: Baseline perplexity and KL metrics from GradualQuantizer
            evaluation_samples: Number of samples for evaluation
            budget_allocations: Per-block budget allocations from importance analysis
            original_model: Reference to original model for KL divergence calculation
            training_batch_size: Batch size for trainable quantization (default: 4)
            training_manager: Reference to training manager for accessing cached data
        """
        self.model = model
        self.tokenizer = tokenizer
        self.min_group_size = min_group_size
        self.max_group_size = max_group_size
        self.baseline_metrics = baseline_metrics or {}
        self.evaluation_samples = evaluation_samples
        self.budget_allocations = budget_allocations or {}
        self.original_model = original_model
        self.training_batch_size = training_batch_size
        self.training_manager = training_manager
        
        # Persistent state for building final quantized model
        self.block_quantizations = {}  # Store best quantization config for each block
        self.quantized_blocks = {}     # Store actual quantized block instances
        self.current_model_state = {}  # Track current state of model blocks
        
        # Get fixed calibration data for consistent evaluation (reusing ImportanceAnalyzer approach)
        self.eval_samples = self._get_evaluation_samples()
        
        print(f"PrecisionOptimizer initialized:")
        print(f"  Budget allocations for {len(self.budget_allocations)} blocks")
        print(f"  Evaluation samples: {self.evaluation_samples}")
        print(f"  Training batch size: {self.training_batch_size}")
        print(f"  Original model reference: {'Available' if original_model else 'Not available'}")
        
    # def progressive_quantize_block(self, block_name: str, cached_data: List, budget_allocation: float) -> Optional[QuantizationConfig]:
    #     """
    #     Progressive quantization algorithm for a single block.
        
    #     Tries RTN first, then training, with decreasing group sizes and bit precision.
    #     Falls back to layer-level recovery if needed.
        
    #     Args:
    #         block_name: Name of block to quantize
    #         cached_data: List of (input_dict, output_tensor) pairs
    #         budget_allocation: Error budget for this block
            
    #     Returns:
    #         Best quantization configuration found, or None if nothing works
    #     """
    #     from .utils import get_module_by_name
        
    #     # Configuration
    #     precision_levels = [8, 4, 2]  # bits
        
    #     # Get the block
    #     block = get_module_by_name(self.model, block_name)
        
    #     # Separate attention and MLP layers
    #     attention_layers, mlp_layers = self._separate_layer_types(block)
        
    #     best_config = None
        
    #     # Phase 1: Try each precision level
    #     for target_bits in precision_levels:
    #         print(f"    Trying {target_bits}-bit quantization...")
            
    #         # Try different group sizes from large to small
    #         group_sizes = self._get_group_size_sequence(self.max_group_size, self.min_group_size)
            
    #         for target_group_size in group_sizes:
    #             print(f"      Group size: {target_group_size}")
                
    #             # Step 1: Try RTN quantization first
    #             rtn_success, rtn_error = self._try_rtn_quantization(
    #                 block, target_bits, target_group_size, cached_data, budget_allocation
    #             )
                
    #             if rtn_success:
    #                 print(f"      ✓ RTN successful (error: {rtn_error:.4f})")
    #                 best_config = QuantizationConfig(target_bits, target_group_size, "RTN", rtn_error)
    #                 continue  # Try even lower precision
                    
    #             # Step 2: RTN failed, try training
    #             print(f"      RTN failed (error: {rtn_error:.4f}), trying training...")
                
    #             train_success, train_error = self._try_trainable_quantization(
    #                 block, target_bits, target_group_size, cached_data, budget_allocation
    #             )
                
    #             if train_success:
    #                 print(f"      ✓ Training successful (error: {train_error:.4f})")
    #                 best_config = QuantizationConfig(target_bits, target_group_size, "Trained", train_error)
    #                 continue  # Try even lower precision
                    
    #             # Step 3: Training failed, try layer-level recovery (only at minimum group size)
    #             if target_group_size == self.min_group_size:
    #                 print(f"      Training failed (error: {train_error:.4f}), trying layer recovery...")
                    
    #                 recovery_success, recovery_error = self._try_layer_recovery(
    #                     block, target_bits, target_group_size, cached_data, budget_allocation
    #                 )
    #             else:
    #                 print(f"      Training failed (error: {train_error:.4f}), skipping layer recovery (group_size={target_group_size} > min={self.min_group_size})")
    #                 recovery_success, recovery_error = False, train_error
                
    #             if recovery_success:
    #                 print(f"      ✓ Layer recovery successful (error: {recovery_error:.4f})")
    #                 best_config = QuantizationConfig(target_bits, target_group_size, "Mixed", recovery_error)
    #                 continue  # Try even lower precision
    #             else:
    #                 print(f"      ✗ Layer recovery failed (error: {recovery_error:.4f})")
    #                 # This precision/group_size combination doesn't work
    #                 if best_config is not None:
    #                     # Return best previous configuration
    #                     print(f"    Using best config: {best_config.bits}-bit, group_size={best_config.group_size}, method={best_config.method}")
    #                     return best_config
    #                 # Continue to next group_size
                    
    #     # Return best configuration found, or None if nothing worked
    #     if best_config is not None:
    #         print(f"  Final config: {best_config.bits}-bit, group_size={best_config.group_size}, method={best_config.method}")
    #         return best_config
    #     else:
    #         print(f"  No quantization possible within budget")
    #         return None
            

    def progressive_quantize_block(self, block_name: str, cached_data: List, budget_allocation: float) -> Optional[QuantizationConfig]:
        """
        Progressive quantization algorithm for a single block.
        
        Tries RTN first, then training, with decreasing group sizes and bit precision.
        Falls back to layer-level recovery if needed.
        
        Args:
            block_name: Name of block to quantize
            cached_data: List of (input_dict, output_tensor) pairs
            budget_allocation: Error budget for this block
            
        Returns:
            Best quantization configuration found, or None if nothing works
        """
        from .utils import get_module_by_name
        
        # Configuration
        precision_levels = [8, 4, 2]  # bits
        
        # Get the block
        block = get_module_by_name(self.model, block_name)
        
        # Separate attention and MLP layers
        attention_layers, mlp_layers = self._separate_layer_types(block)
        
        best_config = None
        
        # Phase 1: Try each precision level
        for target_bits in precision_levels:
            print(f"    Trying {target_bits}-bit quantization...")
            
            # Try different group sizes from large to small
            group_sizes = self._get_group_size_sequence(self.max_group_size, self.min_group_size)
            
            for target_group_size in group_sizes:
                print(f"      Group size: {target_group_size}")
                
                # Step 1: Try RTN quantization first
                rtn_success, rtn_error = self._try_rtn_quantization(
                    block, target_bits, target_group_size, cached_data, budget_allocation
                )
                
                if rtn_success:
                    print(f"      ✓ RTN successful (error: {rtn_error:.4f})")
                    best_config = QuantizationConfig(target_bits, target_group_size, "RTN", rtn_error)
                    continue  # Try even lower precision
                    
                # TEMPORARILY DISABLED: Step 2: RTN failed, try training
                # print(f"      RTN failed (error: {rtn_error:.4f}), trying training...")
                # 
                # train_success, train_error = self._try_trainable_quantization(
                #     block, target_bits, target_group_size, cached_data, budget_allocation
                # )
                # 
                # if train_success:
                #     print(f"      ✓ Training successful (error: {train_error:.4f})")
                #     best_config = QuantizationConfig(target_bits, target_group_size, "Trained", train_error)
                #     continue  # Try even lower precision
                #     
                # # TEMPORARILY DISABLED: Step 3: Training failed, try layer-level recovery
                # if target_group_size == self.min_group_size:
                #     print(f"      Training failed (error: {train_error:.4f}), trying layer recovery...")
                #     
                #     recovery_success, recovery_error = self._try_layer_recovery(
                #         block, target_bits, target_group_size, cached_data, budget_allocation
                #     )
                # else:
                #     print(f"      Training failed (error: {train_error:.4f}), skipping layer recovery (group_size={target_group_size} > min={self.min_group_size})")
                #     recovery_success, recovery_error = False, train_error
                # 
                # if recovery_success:
                #     print(f"      ✓ Layer recovery successful (error: {recovery_error:.4f})")
                #     best_config = QuantizationConfig(target_bits, target_group_size, "Mixed", recovery_error)
                #     continue  # Try even lower precision
                # else:
                #     print(f"      ✗ Layer recovery failed (error: {recovery_error:.4f})")
                
                # RTN failed - check if this was minimum group size
                print(f"      RTN failed (error: {rtn_error:.4f}), skipping training/recovery for testing")
                
                # If this was the minimum group size, stop trying lower bits entirely
                if target_group_size == self.min_group_size:
                    print(f"      Minimum group size ({self.min_group_size}) failed for {target_bits}-bit, stopping precision reduction")
                    if best_config is not None:
                        print(f"    Using best RTN config: {best_config.bits}-bit, group_size={best_config.group_size}, method={best_config.method}")
                        return best_config
                    else:
                        print(f"    No successful configuration found, stopping quantization")
                        return None
                
                # Not minimum group size, continue to next group_size
                if best_config is not None:
                    # Return best previous configuration
                    print(f"    Using best RTN config: {best_config.bits}-bit, group_size={best_config.group_size}, method={best_config.method}")
                    return best_config
                # Continue to next group_size
                    
        # Return best RTN configuration found, or None if nothing worked
        if best_config is not None:
            print(f"  Final RTN config: {best_config.bits}-bit, group_size={best_config.group_size}, method={best_config.method}")
            return best_config
        else:
            print(f"  No RTN quantization possible within budget")
            return None
            
    def _separate_layer_types(self, block):
        """Separate attention and MLP layers in a transformer block."""
        attention_layers = []
        mlp_layers = []
        
        for name, module in block.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Simple heuristic to identify layer types
                if any(keyword in name.lower() for keyword in ['attn', 'attention', 'query', 'key', 'value']):
                    attention_layers.append((name, module))
                elif any(keyword in name.lower() for keyword in ['mlp', 'fc', 'linear', 'feed_forward']):
                    mlp_layers.append((name, module))
                else:
                    # Default to MLP if unclear
                    mlp_layers.append((name, module))
                    
        return attention_layers, mlp_layers
        
    def _get_group_size_sequence(self, max_size, min_size):
        """Generate sequence of group sizes from max to min."""
        sizes = []
        current = max_size
        while current >= min_size:
            sizes.append(current)
            current = current // 2
        return sizes
        
    def _try_rtn_quantization(self, block, bits, group_size, cached_data, budget):
        """Try RTN quantization with given parameters."""
        from .base_wrapper import RTNWrapper
        from .utils import get_module_by_name, set_module_by_name
        
        # Get block name from model
        block_name = self._find_block_name(block)
        if not block_name:
            print(f"      Could not find block name in model")
            return False, float('inf')
        
        print(f"      Applying RTN {bits}-bit quantization to {block_name}")
        
        try:
            # Get current block state (original or previously quantized)
            current_block = get_module_by_name(self.model, block_name)
            
            # Create RTN wrapper for this block
            rtn_wrapper = RTNWrapper(
                wrapped_block=current_block,
                bits=bits,
                group_size=group_size,
                storage_device="cpu",  # Use CPU storage for memory efficiency
                block_name=block_name
            )
            
            # Replace block with wrapper in model
            set_module_by_name(self.model, block_name, rtn_wrapper)
            
            # Enable quantization for this block
            rtn_wrapper.enable()
            
            # Evaluate model performance with this block quantized
            performance = self.evaluate_block_performance(block_name)
            
            print(f"        RTN result: PPL increase = {performance['ppl_increase']:.4f}")
            print(f"        Combined sensitivity = {performance['current_combined_sensitivity']:.4f}, threshold = {performance['threshold_combined_sensitivity']:.4f}")
            
            if performance['under_threshold']:
                # Save this quantization configuration
                config = QuantizationConfig(
                    bits=bits,
                    group_size=group_size,
                    method="RTN",
                    error_metric=performance['ppl_increase'],
                    memory_savings=self._estimate_memory_savings(bits, group_size)
                )
                
                # Store quantized wrapper if it's better than existing
                self._save_block_quantization_if_better(block_name, config, rtn_wrapper)
                
                return True, performance['ppl_increase']
            else:
                # RTN failed - restore previous state
                self._restore_block_state(block_name, current_block)
                rtn_wrapper.cleanup()
                
                return False, performance['ppl_increase']
            
        except Exception as e:
            print(f"        RTN quantization failed: {e}")
            # Ensure cleanup on error
            try:
                if 'rtn_wrapper' in locals():
                    rtn_wrapper.cleanup()
                    self._restore_block_state(block_name, current_block)
            except:
                pass
            return False, float('inf')
            
    def _find_block_name(self, target_block):
        """Find the name of a block in the model."""
        for name, module in self.model.named_modules():
            if module is target_block:
                return name
        return None
        
    def _save_block_quantization_if_better(self, block_name: str, config: QuantizationConfig, quantized_block):
        """Save quantization configuration only if it uses less memory than current saved config."""
        # Calculate memory usage for new config
        new_memory = self._calculate_block_memory_usage(quantized_block, config)
        
        if block_name in self.block_quantizations:
            existing_config = self.block_quantizations[block_name]
            existing_block = self.quantized_blocks[block_name]
            existing_memory = self._calculate_block_memory_usage(existing_block, existing_config)
            
            # Compare memory usage
            if new_memory < existing_memory:
                compression_improvement = existing_memory - new_memory
                print(f"        ⚡ Better compression: {existing_memory:.1f}MB → {new_memory:.1f}MB (-{compression_improvement:.1f}MB)")
                
                # Clean up previous quantized block
                if hasattr(self.quantized_blocks[block_name], 'cleanup'):
                    self.quantized_blocks[block_name].cleanup()
                    
                # Save new better config
                self.block_quantizations[block_name] = config
                self.quantized_blocks[block_name] = quantized_block
                self.current_model_state[block_name] = 'quantized'
                
                print(f"        ✓ Saved improved {config.method} {config.bits}-bit for {block_name}")
                return True
            else:
                memory_difference = new_memory - existing_memory
                print(f"        → Keeping existing config: {existing_memory:.1f}MB < {new_memory:.1f}MB (+{memory_difference:.1f}MB)")
                
                # Clean up current attempt since we're not using it
                if hasattr(quantized_block, 'cleanup'):
                    quantized_block.cleanup()
                return False
        else:
            # First config for this block
            self.block_quantizations[block_name] = config
            self.quantized_blocks[block_name] = quantized_block
            self.current_model_state[block_name] = 'quantized'
            
            print(f"        ✓ Saved {config.method} {config.bits}-bit for {block_name} ({new_memory:.1f}MB)")
            return True
    
    def _save_block_quantization(self, block_name: str, config: QuantizationConfig, quantized_block):
        """Legacy function - redirects to memory-based saving."""
        return self._save_block_quantization_if_better(block_name, config, quantized_block)
        
    def _restore_block_state(self, block_name: str, previous_block):
        """Restore block to previous state."""
        from .utils import set_module_by_name
        set_module_by_name(self.model, block_name, previous_block)
        
    def _estimate_memory_savings(self, bits: int, group_size: int) -> float:
        """Simple memory savings: lower bits = less memory."""
        return (1 - bits / 32.0) * 100  # Compare to float32
        
    def _calculate_block_memory_usage(self, quantized_block, config: QuantizationConfig) -> float:
        """Calculate actual memory usage of a quantized block in MB."""
        total_memory = 0.0
        
        for name, module in quantized_block.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Get layer name relative to block
                layer_name = name
                
                if config.layer_exceptions and layer_name in config.layer_exceptions:
                    # Frozen layer: higher precision but minimum group size
                    memory = self._calculate_layer_memory(
                        module,
                        bits=8,  # Frozen layers use higher precision
                        group_size=self.min_group_size
                    )
                else:
                    # Quantized layer: uses config settings
                    memory = self._calculate_layer_memory(
                        module,
                        bits=config.bits,
                        group_size=config.group_size
                    )
                total_memory += memory
                
        return total_memory
        
    def _calculate_layer_memory(self, layer: torch.nn.Linear, bits: int, group_size: int) -> float:
        """Calculate memory usage for a single layer in MB."""
        out_features, in_features = layer.weight.shape
        
        # Quantized weights memory
        values_per_int32 = 32 // bits
        total_elements = out_features * in_features
        packed_elements = (total_elements + values_per_int32 - 1) // values_per_int32  # Ceiling division
        quantized_weight_memory = packed_elements * 4  # int32 = 4 bytes
        
        # Scale and zero_point tensors
        num_groups = (in_features + group_size - 1) // group_size  # Ceiling division
        scale_memory = out_features * num_groups * 4  # float32 scales
        zp_memory = out_features * num_groups * 4     # int32 zero points
        
        # Bias (if present)
        bias_memory = 0
        if layer.bias is not None:
            bias_memory = out_features * 4  # float32 bias
            
        total_memory_bytes = quantized_weight_memory + scale_memory + zp_memory + bias_memory
        return total_memory_bytes / (1024 * 1024)  # Convert to MB
        
        
    def freeze_layers(self, block_name: str, layer_names: List[str]):
        """Mark specific layers as frozen (won't be quantized further)."""
        if block_name not in self.block_quantizations:
            return
            
        config = self.block_quantizations[block_name]
        for layer_name in layer_names:
            config.layer_exceptions[layer_name] = "frozen"
        
        print(f"        Froze {len(layer_names)} layers in {block_name}")
        
    def is_layer_frozen(self, block_name: str, layer_name: str) -> bool:
        """Check if a layer is frozen from further quantization."""
        if block_name not in self.block_quantizations:
            return False
        
        config = self.block_quantizations[block_name]
        return config.layer_exceptions.get(layer_name) == "frozen"
        
    def evaluate_block_performance(self, block_name: str) -> Dict[str, float]:
        """
        Evaluate if block achieves performance under threshold.
        Simple: evaluate quantized model, subtract baseline, calculate combined sensitivity.
        
        Returns:
            Dict with evaluation results and threshold check
        """
        from ..utils.evaluation import calculate_perplexity, calculate_kl_divergence
        
        # Get baseline metrics (already calculated in GradualQuantizer)
        baseline_ppl = self.baseline_metrics.get('perplexity', 0.0)
        baseline_kl = self.baseline_metrics.get('kl_divergence', 0.0)
        
        # Evaluate current quantized model (with one block quantized)
        current_ppl = calculate_perplexity(
            model=self.model,
            tokenizer=self.tokenizer,
            eval_samples=self.evaluation_samples,
            verbose=False
        )
        
        # Calculate KL divergence if we have original model reference
        current_kl = baseline_kl  # Default to baseline if no original model
        if self.original_model is not None:
            _, current_kl = calculate_kl_divergence(
                original_model=self.original_model,
                quantized_model=self.model,
                tokenizer=self.tokenizer,
                eval_samples=self.evaluation_samples,
                verbose=False
            )
        
        # Calculate degradation
        ppl_increase = current_ppl - baseline_ppl
        kl_increase = current_kl - baseline_kl
        
        # Calculate combined sensitivity for current state (using same weights as in budget allocation)
        from .importance_analyzer import ImportanceAnalyzer
        analyzer = ImportanceAnalyzer(self.model, self.tokenizer, evaluation_samples=self.evaluation_samples)
        
        current_combined_sensitivity = analyzer.calculate_combined_sensitivity(
            ppl_increase=ppl_increase,
            kl_increase=kl_increase,
            max_ppl_increase=allocated_budget,  # Use allocated budget as max
            max_kl_increase=self.baseline_metrics['kl_token_budget']  # Use from baseline
        )
        
        # Get budget threshold for this block (already calculated in importance analysis)
        block_budget = self.budget_allocations.get(block_name, {})
        threshold_combined_sensitivity = block_budget.get('combined_sensitivity', float('inf'))
        
        # Check if under threshold
        under_threshold = current_combined_sensitivity <= threshold_combined_sensitivity
        
        return {
            'current_ppl': current_ppl,
            'baseline_ppl': baseline_ppl,
            'ppl_increase': ppl_increase,
            'current_kl': current_kl,
            'baseline_kl': baseline_kl,
            'kl_increase': kl_increase,
            'current_combined_sensitivity': current_combined_sensitivity,
            'threshold_combined_sensitivity': threshold_combined_sensitivity,
            'under_threshold': under_threshold
        }
        
    def _try_trainable_quantization(self, block, bits, group_size, cached_data, budget):
        """Try trainable quantization with given parameters using existing Bitween training infrastructure."""
        from ..wrapper import wrap_model_for_training, train_individual_wrapper, finalize_wrapped_model
        from .utils import get_module_by_name, set_module_by_name
        
        block_name = self._find_block_name(block)
        if not block_name:
            print(f"        Could not find block name in model")
            return False, float('inf')
        
        print(f"        Training {block_name} with {len(cached_data)} samples...")
        
        try:
            # Get current block state
            current_block = get_module_by_name(self.model, block_name)
            
            # Step 1: Wrap the block for training
            wrapped_info = wrap_model_for_training(
                model=self.model,
                block_names=[block_name],  # Only this block
                enable_minmax_tuning=True,
                bits=bits,
                group_size=group_size,
                ignore_layers=set()  # Don't ignore any layers within the block
            )
            
            if block_name not in wrapped_info:
                print(f"        Failed to wrap {block_name}")
                return False, float('inf')
            
            wrapper_info = wrapped_info[block_name]
            
            # Step 2: Convert cached_data format for training
            # cached_data is List[(input_dict, output_tensor)]
            block_inputs = []
            for input_dict, output_tensor in cached_data:
                # Extract the input tensor from the input_dict
                input_tensor = input_dict.get('hidden_states', input_dict.get('input', None))
                if input_tensor is not None:
                    block_inputs.append(input_tensor)
            
            if not block_inputs:
                print(f"        No valid inputs found in cached data")
                return False, float('inf')
            
            # Step 3: Train the wrapped block (automatically handles frozen layers)
            result = self._train_wrapper_respecting_frozen_layers(
                module_name=block_name,
                wrapped_module=wrapper_info['wrapped_module'],
                block_inputs=block_inputs,
                iters=1,  # Number of training epochs
                lr=1e-4,  # Learning rate 
                batch_size=self.training_batch_size,
                is_single_layer=wrapper_info['is_single_layer']
            )
            
            # Step 4: Apply best parameters (automatically skips frozen layers)
            if not wrapper_info['is_single_layer']:
                self._apply_best_params_respecting_frozen_layers(block_name, wrapper_info['wrapped_module'])
            
            # Step 5: Evaluate performance
            performance = self.evaluate_block_performance(block_name)
            
            print(f"        Training result: PPL increase = {performance['ppl_increase']:.4f}")
            print(f"        Combined sensitivity = {performance['current_combined_sensitivity']:.4f}, threshold = {performance['threshold_combined_sensitivity']:.4f}")
            
            if performance['under_threshold']:
                # Step 6: Convert to final quantized form and save
                finalized_block = finalize_wrapped_model(self.model, {block_name: wrapper_info})
                finalized_block_module = get_module_by_name(finalized_block, block_name)
                
                # Replace in our model
                set_module_by_name(self.model, block_name, finalized_block_module)
                
                # Save configuration
                config = QuantizationConfig(
                    bits=bits,
                    group_size=group_size,
                    method="Trained",
                    error_metric=performance['ppl_increase'],
                    training_metadata={
                        'epochs': 3,
                        'lr': 1e-4,
                        'batch_size': self.training_batch_size,
                        'samples': len(block_inputs)
                    },
                    memory_savings=self._estimate_memory_savings(bits, group_size)
                )
                
                self._save_block_quantization_if_better(block_name, config, finalized_block_module)
                
                return True, performance['ppl_increase']
            else:
                # Training failed - restore previous state
                self._restore_block_state(block_name, current_block)
                
                return False, performance['ppl_increase']
                
        except Exception as e:
            print(f"        Training quantization failed: {e}")
            # Restore previous state on error
            try:
                self._restore_block_state(block_name, current_block)
            except:
                pass
            return False, float('inf')
        
    def _analyze_layer_quality_impact(self, block):
        """Analyze quality impact of reverting each layer to previously saved quantized state."""
        layer_impacts = []
        
        block_name = self._find_block_name(block)
        if not block_name:
            return []
            
        # Check if we have a previously saved working quantized block to revert to
        if block_name not in self.block_quantizations:
            print(f"        No previously saved quantization for {block_name} - cannot do layer recovery")
            return []
            
        saved_quantized_block = self.quantized_blocks.get(block_name)
        if saved_quantized_block is None:
            print(f"        No saved quantized block available for {block_name}")
            return []
        
        # Use existing function to separate layer types
        attention_layers, mlp_layers = self._separate_layer_types(block)
        
        # Test attention layers first (usually more sensitive), then MLP layers
        all_layers = attention_layers + mlp_layers
        
        print(f"        Analyzing {len(attention_layers)} attention + {len(mlp_layers)} MLP layers...")
        print(f"        Reverting to saved {self.block_quantizations[block_name].method} {self.block_quantizations[block_name].bits}-bit quantization")
        
        # Get cached data from training manager
        cached_data = self._get_cached_data_for_block(block_name)
        if not cached_data:
            print(f"        No cached data available for {block_name}")
            return []
        
        # Get baseline outputs with current failed quantization (calculate once)
        baseline_outputs = self._get_block_outputs_on_cached_samples(block_name, cached_data[:self.evaluation_samples])
        
        for layer_name, layer_module in all_layers:
            # Skip if layer is already frozen
            if self.is_layer_frozen(block_name, layer_name):
                continue
                
            print(f"          Testing impact of reverting {layer_name}...")
            
            try:
                # Measure actual impact by reverting layer to saved quantized state
                quality_improvement = self._measure_layer_reversion_impact(block_name, layer_name, cached_data, baseline_outputs, saved_quantized_block)
                
                layer_type = "attention" if any(keyword in layer_name.lower() 
                                               for keyword in ['attn', 'attention', 'query', 'key', 'value']) else "mlp"
                
                layer_impacts.append({
                    'layer_name': layer_name,
                    'layer_type': layer_type,
                    'quality_improvement': quality_improvement,
                })
                
                print(f"            {layer_name} ({layer_type}): improvement={quality_improvement:.4f}")
                
            except Exception as e:
                print(f"            Failed to analyze {layer_name}: {e}")
                continue
        
        # Sort by quality improvement (most impactful layers first)
        layer_impacts.sort(key=lambda x: x['quality_improvement'], reverse=True)
        
        print(f"        Layer analysis complete. Top layer: {layer_impacts[0]['layer_name'] if layer_impacts else 'None'}")
        return layer_impacts
        
    def _get_cached_data_for_block(self, block_name):
        """Get cached training data for a specific block from the training manager."""
        try:
            if not self.training_manager:
                print(f"        Warning: No training manager available for cached data")
                return []
            
            # Get the wrapper for this block
            if hasattr(self.training_manager, 'block_wrappers') and block_name in self.training_manager.block_wrappers:
                wrapper = self.training_manager.block_wrappers[block_name]
                if hasattr(wrapper, 'get_all_cached_data'):
                    cached_data = wrapper.get_all_cached_data()
                    print(f"        Retrieved {len(cached_data)} cached samples for {block_name}")
                    return cached_data
            
            print(f"        No cached data found for {block_name}")
            return []
            
        except Exception as e:
            print(f"        Error getting cached data for {block_name}: {e}")
            return []
        
    def _measure_layer_reversion_impact(self, block_name, layer_name, cached_data, baseline_outputs, saved_quantized_block):
        """Measure KL divergence reduction from reverting a layer to saved quantized state."""
        try:
            # Use same samples as baseline
            validation_samples = cached_data[:self.evaluation_samples]
            
            if not validation_samples:
                print(f"            No validation samples available for {layer_name}")
                return 0.0
            
            # Get outputs with layer reverted to saved quantized state
            reverted_outputs = self._get_block_outputs_with_reverted_layer_from_saved(block_name, layer_name, validation_samples, saved_quantized_block)
            
            if reverted_outputs is None:
                return 0.0
            
            # Calculate KL divergence between reverted and current failed quantization outputs
            kl_divergence = self._calculate_kl_divergence_between_outputs(reverted_outputs, baseline_outputs)
            
            # Higher KL divergence = more impact from this layer
            return kl_divergence
            
        except Exception as e:
            print(f"            Failed to measure impact for {layer_name}: {e}")
            return 0.0
            
    def _get_block_outputs_on_cached_samples(self, block_name, cached_samples):
        """Get block outputs using cached input samples."""
        from .utils import get_module_by_name
        
        try:
            if not cached_samples:
                return None
            
            outputs = []
            
            # Get outputs from current block
            with torch.no_grad():
                for input_dict, _ in cached_samples:
                    # Extract input tensor
                    input_tensor = input_dict.get('hidden_states', input_dict.get('input', None))
                    if input_tensor is None:
                        continue
                    
                    # Get current block
                    block = get_module_by_name(self.model, block_name)
                    
                    # Forward through block
                    output = block(input_tensor)
                    if isinstance(output, tuple):
                        output = output[0]
                    
                    outputs.append(output.detach())
            
            return outputs if outputs else None
            
        except Exception as e:
            print(f"            Error getting block outputs: {e}")
            return None
        
    def _get_block_outputs_with_reverted_layer_from_saved(self, block_name, layer_name, cached_samples, saved_quantized_block):
        """Get block outputs with one layer temporarily reverted to saved quantized state."""
        from .utils import get_module_by_name, set_module_by_name
        
        try:
            # Get current layer
            current_layer = get_module_by_name(self.model, f"{block_name}.{layer_name}")
            
            # Get saved quantized layer
            saved_layer = get_module_by_name(saved_quantized_block, layer_name)
            
            if saved_layer is None:
                print(f"            Could not find saved layer for {layer_name}")
                return None
            
            # Temporarily replace with saved quantized layer
            set_module_by_name(self.model, f"{block_name}.{layer_name}", saved_layer)
            
            # Get outputs with reverted layer
            reverted_outputs = self._get_block_outputs_on_cached_samples(block_name, cached_samples)
            
            # Restore current layer
            set_module_by_name(self.model, f"{block_name}.{layer_name}", current_layer)
            
            return reverted_outputs
            
        except Exception as e:
            print(f"            Error getting outputs with reverted layer {layer_name}: {e}")
            # Make sure to restore current layer on error
            try:
                set_module_by_name(self.model, f"{block_name}.{layer_name}", current_layer)
            except:
                pass
            return None
            
    def _calculate_kl_divergence_between_outputs(self, outputs1, outputs2):
        """Calculate KL divergence between two sets of block outputs."""
        import torch.nn.functional as F
        
        try:
            if not outputs1 or not outputs2 or len(outputs1) != len(outputs2):
                return 0.0
            
            total_kl = 0.0
            total_tokens = 0
            
            for out1, out2 in zip(outputs1, outputs2):
                # Ensure same shape
                min_shape = [min(s1, s2) for s1, s2 in zip(out1.shape, out2.shape)]
                out1_trimmed = out1[:min_shape[0], :min_shape[1], :min_shape[2]]
                out2_trimmed = out2[:min_shape[0], :min_shape[1], :min_shape[2]]
                
                # Calculate KL divergence (treat as logits)
                log_prob1 = F.log_softmax(out1_trimmed, dim=-1)
                prob2 = F.softmax(out2_trimmed, dim=-1)
                
                # Per-token KL divergence
                per_token_kl = F.kl_div(log_prob1, prob2, reduction='none', log_target=False)
                per_token_kl = per_token_kl.sum(dim=-1)  # sum over vocab dim
                
                total_kl += per_token_kl.sum().item()
                total_tokens += per_token_kl.numel()
            
            avg_kl = total_kl / total_tokens if total_tokens > 0 else 0.0
            return avg_kl
            
        except Exception as e:
            print(f"            Error calculating KL divergence: {e}")
            return 0.0
            
    def _create_original_layer_from_quantized(self, quantized_layer):
        """Create original torch.nn.Linear layer from quantized wrapper."""
        try:
            # Check if this is a wrapper with original weights
            if hasattr(quantized_layer, 'orig_weight'):
                # Extract original weights and bias from wrapper
                original_weight = quantized_layer.orig_weight
                original_bias = getattr(quantized_layer, 'orig_bias', None)
                
                # Create regular Linear layer
                original_layer = torch.nn.Linear(
                    in_features=original_weight.shape[1],
                    out_features=original_weight.shape[0],
                    bias=original_bias is not None,
                    device=original_weight.device,
                    dtype=original_weight.dtype
                )
                
                # Copy weights
                original_layer.weight.data = original_weight.clone()
                if original_bias is not None:
                    original_layer.bias.data = original_bias.clone()
                
                return original_layer
                
            elif isinstance(quantized_layer, torch.nn.Linear):
                # Already a regular layer, return as-is
                return quantized_layer
                
            else:
                print(f"            Unknown layer type: {type(quantized_layer)}")
                return None
                
        except Exception as e:
            print(f"            Failed to create original layer: {e}")
            return None
        
    def _find_minimum_recovery_set(self, block_name, sorted_layer_impacts, budget):
        """Find minimum set of layers to revert using exponential search.
        
        Algorithm:
        1. Try reverting 1 layer → evaluate whole model → pass/fail?
        2. Try reverting 2 layers → evaluate whole model → pass/fail?
        3. Try reverting 4 layers → evaluate whole model → pass/fail?
        4. Continue with 8, 16, etc. until success or all layers tested
        5. Return minimum working set
        """
        print(f"        Starting exponential search for minimum recovery set...")
        
        if not sorted_layer_impacts:
            print(f"        No layers to recover")
            return []
        
        # Exponential search: 1, 2, 4, 8, 16, ...
        recovery_count = 1
        best_working_set = None
        
        while recovery_count <= len(sorted_layer_impacts):
            # Get top N layers to revert
            recovery_set = sorted_layer_impacts[:recovery_count]
            layer_names = [layer['layer_name'] for layer in recovery_set]
            
            print(f"          Testing recovery set of {recovery_count} layers: {layer_names[:3]}{'...' if len(layer_names) > 3 else ''}")
            
            try:
                # Temporarily revert these layers and evaluate whole model
                meets_budget = self._test_recovery_set(block_name, recovery_set)
                
                if meets_budget:
                    print(f"          ✓ Recovery set of {recovery_count} layers meets budget")
                    best_working_set = recovery_set
                    break  # Found minimum working set
                else:
                    print(f"          ✗ Recovery set of {recovery_count} layers fails budget")
                    
            except Exception as e:
                print(f"          Error testing recovery set of {recovery_count} layers: {e}")
            
            # Exponential progression: 1 → 2 → 4 → 8 → 16...
            recovery_count *= 2
        
        if best_working_set:
            print(f"        Found minimum recovery set: {len(best_working_set)} layers")
            return best_working_set
        else:
            # If no subset works, try all layers as last resort
            print(f"        No subset works, trying all {len(sorted_layer_impacts)} layers")
            if self._test_recovery_set(block_name, sorted_layer_impacts):
                return sorted_layer_impacts
            else:
                print(f"        Complete layer recovery failed")
                return []
                
    def _test_recovery_set(self, block_name, recovery_set):
        """Test if reverting a set of layers to previous saved quantized state meets the quality budget.
        
        Returns True if the recovery set meets budget, False otherwise.
        """
        from .utils import get_module_by_name, set_module_by_name
        
        if not recovery_set:
            return False
            
        # Check if we have a previously saved working quantized block
        if block_name not in self.block_quantizations:
            print(f"            No previously saved quantization for {block_name}")
            return False
            
        saved_quantized_block = self.quantized_blocks.get(block_name)
        if saved_quantized_block is None:
            print(f"            No saved quantized block available for {block_name}")
            return False
        
        # Store current block state for restoration
        current_block = get_module_by_name(self.model, block_name)
        current_layers = {}
        
        try:
            # Step 1: Revert specified layers to saved quantized state
            for layer_info in recovery_set:
                layer_name = layer_info['layer_name']
                full_layer_path = f"{block_name}.{layer_name}"
                
                # Store current layer
                current_layer = get_module_by_name(self.model, full_layer_path)
                current_layers[full_layer_path] = current_layer
                
                # Get corresponding layer from saved quantized block
                try:
                    saved_layer = get_module_by_name(saved_quantized_block, layer_name)
                    if saved_layer is None:
                        print(f"            Layer {layer_name} not found in saved block")
                        continue
                        
                    # Replace with saved quantized layer
                    set_module_by_name(self.model, full_layer_path, saved_layer)
                    print(f"            Reverted {layer_name} to saved quantized state")
                    
                except Exception as e:
                    print(f"            Could not revert {layer_name}: {e}")
                    continue
            
            # Step 2: Evaluate whole model performance with reverted layers
            performance = self.evaluate_block_performance(block_name)
            
            # Step 3: Check if meets budget
            meets_budget = performance['under_threshold']
            
            print(f"            Combined sensitivity: {performance['current_combined_sensitivity']:.4f} (threshold: {performance['threshold_combined_sensitivity']:.4f})")
            
            return meets_budget
            
        except Exception as e:
            print(f"            Error testing recovery set: {e}")
            return False
            
        finally:
            # Step 4: Always restore current layers
            try:
                for full_layer_path, current_layer in current_layers.items():
                    set_module_by_name(self.model, full_layer_path, current_layer)
            except Exception as e:
                print(f"            Error restoring current layers: {e}")
        
    def _retrain_mixed_precision_block(self, block_name, recovery_set, cached_data, target_bits, target_group_size):
        """Retrain block with mixed precision (some layers frozen from previous saved block).
        
        Algorithm:
        1. Copy frozen layers from previously saved quantized block (already trained)
        2. Wrap block for training with target quantization (bits, group_size from loop)
        3. Replace frozen layers with saved versions and freeze them from training
        4. Train only non-frozen layers while keeping frozen layers fixed
        """
        from ..wrapper import wrap_model_for_training
        from .utils import get_module_by_name, set_module_by_name
        
        print(f"          Retraining {block_name} with {len(recovery_set)} layers frozen from saved block...")
        print(f"          Target quantization for non-frozen layers: {target_bits}-bit, group_size={target_group_size}")
        
        try:
            # Get current block state
            current_block = get_module_by_name(self.model, block_name)
            
            # Get saved quantized block (source of frozen layers)
            saved_quantized_block = self.quantized_blocks.get(block_name)
            if saved_quantized_block is None:
                print(f"          No saved quantized block available for frozen layers")
                return current_block
            
            # Get frozen layer names
            frozen_layer_names = [layer_info['layer_name'] for layer_info in recovery_set]
            
            # Step 1: Wrap the block for training with target quantization from loop
            wrapped_info = wrap_model_for_training(
                model=self.model,
                block_names=[block_name],  # Only this block
                enable_minmax_tuning=True,
                bits=target_bits,  # Use bits from quantization loop
                group_size=target_group_size,  # Use group_size from quantization loop
                ignore_layers=set()  # Don't ignore any layers in wrapping
            )
            
            if block_name not in wrapped_info:
                print(f"          Failed to wrap {block_name} for mixed precision training")
                return current_block
            
            wrapper_info = wrapped_info[block_name]
            wrapped_block = wrapper_info['wrapped_module']
            
            # Step 2: Replace frozen layers with saved quantized versions
            for layer_name in frozen_layer_names:
                try:
                    # Get saved quantized layer (already trained)
                    saved_layer = get_module_by_name(saved_quantized_block, layer_name)
                    if saved_layer is None:
                        print(f"            Warning: Layer {layer_name} not found in saved block")
                        continue
                    
                    # Replace wrapped layer with saved quantized layer
                    set_module_by_name(wrapped_block, layer_name, saved_layer)
                    print(f"            Copied saved quantized layer: {layer_name}")
                    
                except Exception as e:
                    print(f"            Failed to copy saved layer {layer_name}: {e}")
                    continue
            
            # Step 3: Convert cached_data format for training
            block_inputs = []
            for input_dict, output_tensor in cached_data:
                input_tensor = input_dict.get('hidden_states', input_dict.get('input', None))
                if input_tensor is not None:
                    block_inputs.append(input_tensor)
            
            if not block_inputs:
                print(f"          No valid inputs found in cached data")
                return current_block
            
            # Step 4: Train with frozen layers excluded from optimizer (use unified function)
            result = self._train_wrapper_respecting_frozen_layers(
                module_name=block_name,
                wrapped_module=wrapped_block,
                block_inputs=block_inputs,
                iters=1,  # Number of training epochs
                lr=1e-4,  # Learning rate 
                batch_size=self.training_batch_size,
                is_single_layer=wrapper_info['is_single_layer']
            )
            
            return wrapped_block
            
        except Exception as e:
            print(f"          Mixed precision retraining failed: {e}")
            return current_block
        
    def _apply_layer_recovery_config(self, block_name, config, recovery_set):
        """Apply mixed precision configuration and mark problematic layers as frozen.
        
        Algorithm:
        1. Update QuantizationConfig with layer exceptions
        2. Mark recovered layers as frozen for future attempts
        3. Save memory-optimized config (higher precision but min group size)
        
        TODO: Add layer_exceptions to QuantizationConfig
        TODO: Set frozen layers to: 8-bit precision + min_group_size
        TODO: Update freeze_layer tracking for this block
        """
        print(f"          Applying layer recovery config to {block_name}")
        
        # Mark layers as frozen
        for layer_info in recovery_set:
            layer_name = layer_info['layer_name']
            self.freeze_layers(block_name, [layer_name])
            print(f"            Froze layer: {layer_name}")
        
        # Update config with layer exceptions
        config.layer_exceptions = config.layer_exceptions or {}
        for layer_info in recovery_set:
            config.layer_exceptions[layer_info['layer_name']] = {
                'reason': 'quality_recovery',
                'original_bits': config.bits,
                'recovery_bits': 8,
                'recovery_group_size': self.min_group_size
            }
        
        return config
    
    def _try_layer_recovery(self, block, bits, group_size, cached_data, budget):
        """Try layer-level recovery for failed quantization.
        
        LAYER RECOVERY ALGORITHM:
        ========================
        Purpose: Get block to pass quality budget when normal quantization fails
        Constraint: ONLY run at minimum group size (last resort)
        
        Steps:
        1. Analyze Impact: Test each layer - revert to 8-bit, measure perplexity improvement
        2. Exponential Search: Try reverting 1→2→4→8 layers until quality budget met
        3. Mixed Training: Retrain with some layers 8-bit, others at target precision
        4. Freeze Layers: Mark problematic layers as frozen (won't quantize further)
        5. Save Config: Mixed precision config (some 4-bit, some 8-bit)
        
        Quality Focus: Ignore memory usage - that's optimized later globally
        """
        # CRITICAL: Only run at minimum group size
        if group_size > self.min_group_size:
            print(f"        Skipping layer recovery: group_size={group_size} > min={self.min_group_size}")
            return False, float('inf')
        
        block_name = self._find_block_name(block)
        print(f"        Starting layer recovery for {block_name}: {bits}-bit at min group_size={group_size}")
        
        try:
            # Phase 1: Analyze which layers impact quality most (sorted by impact)
            layer_impacts = self._analyze_layer_quality_impact(block, bits, group_size)
            
            if not layer_impacts:
                print(f"        No layers found for recovery analysis")
                return False, float('inf')
            
            # Phase 2: Find minimum set of layers to revert (exponential search)
            recovery_set = self._find_minimum_recovery_set(block_name, layer_impacts, budget)
            
            if not recovery_set:
                print(f"        No recovery set found")
                return False, float('inf')
            
            # Phase 3: Retrain with mixed precision (some layers frozen from saved block)
            mixed_precision_block = self._retrain_mixed_precision_block(block_name, recovery_set, cached_data, bits, group_size)
            
            # Phase 4: Apply recovery configuration and freeze problematic layers
            config = QuantizationConfig(
                bits=bits,
                group_size=group_size,
                method="Mixed",
                error_metric=0.0,  # Will be updated after evaluation
                memory_savings=self._estimate_memory_savings(bits, group_size)
            )
            
            recovery_config = self._apply_layer_recovery_config(block_name, config, recovery_set)
            
            # Phase 5: Final evaluation
            performance = self.evaluate_block_performance(block_name)
            recovery_config.error_metric = performance['ppl_increase']
            
            if performance['under_threshold']:
                # Save the mixed precision configuration
                self._save_block_quantization_if_better(block_name, recovery_config, mixed_precision_block)
                print(f"        ✓ Layer recovery successful: {len(recovery_set)} layers frozen")
                return True, performance['ppl_increase']
            else:
                print(f"        ✗ Layer recovery failed: still over budget")
                return False, performance['ppl_increase']
                
        except Exception as e:
            print(f"        Layer recovery failed with error: {e}")
            return False, float('inf')
        
        
    def _get_evaluation_samples(self):
        """Get fixed calibration data for consistent evaluation (reusing ImportanceAnalyzer approach)."""
        from ..calib_dataset import get_calibration_dataset
        
        # Get small, fixed dataset for testing (same as ImportanceAnalyzer._get_fixed_calibration_data)
        calib_dataset = get_calibration_dataset(
            dataset_name="pile-10k",
            tokenizer=self.tokenizer,
            seqlen=512,
            nsamples=self.evaluation_samples,
            seed=123  # Same seed as ImportanceAnalyzer
        )
        
        # Get the tokenized samples
        samples = calib_dataset.get_samples(num_samples=self.evaluation_samples)
        return samples
        
    def get_block_budget_info(self, block_name: str) -> Dict:
        """Get budget allocation information for a specific block."""
        block_budget = self.budget_allocations.get(block_name, {})
        
        return {
            'allocated_ppl_budget': block_budget.get('allocated_ppl_budget', 0.0),
            'allocated_kl_budget': block_budget.get('allocated_kl_budget', 0.0),
            'ppl_budget_percent': block_budget.get('ppl_budget_percent', 0.0),
            'kl_budget_percent': block_budget.get('kl_budget_percent', 0.0),
            'ppl_sensitivity': block_budget.get('ppl_sensitivity', 0.0),
            'kl_sensitivity': block_budget.get('kl_sensitivity', 0.0),
            'combined_sensitivity': block_budget.get('combined_sensitivity', 0.0)
        }
         
    def _get_model_logits(self):
        """Get current model logits on evaluation samples."""
        self.model.eval()
        device = next(self.model.parameters()).device
        
        current_logits = []
        
        with torch.no_grad():
            for sample in self.eval_samples:
                input_ids = sample["input_ids"].unsqueeze(0).to(device)
                
                # Get current state logits
                outputs = self.model(input_ids)
                logits = outputs.logits  # shape: [1, seq_len, vocab]
                current_logits.append(logits)
                
        return current_logits
        
    def _evaluate_block_error(self, block, cached_data):
        """Evaluate reconstruction error on cached data."""
        # For now, evaluate full model perplexity as proxy for block quality
        # TODO: Could implement more targeted block-level evaluation
        current_ppl = self._evaluate_model_perplexity()
        baseline_ppl = self.baseline_metrics.get('perplexity', current_ppl)
        
        # Return perplexity increase as error metric
        ppl_increase = current_ppl - baseline_ppl
        return max(0, ppl_increase)  # Ensure non-negative
        
    def apply_all_quantizations(self):
        """Apply all saved quantization configurations to build final quantized model."""
        print(f"\nApplying all quantizations to build final model...")
        print(f"Total blocks to quantize: {len(self.block_quantizations)}")
        
        total_memory_savings = 0.0
        quantization_summary = []
        
        for block_name, config in self.block_quantizations.items():
            if block_name in self.quantized_blocks:
                # Block is already quantized with the saved config
                print(f"  ✓ {block_name}: {config.method} {config.bits}-bit (group_size={config.group_size})")
                total_memory_savings += config.memory_savings
                quantization_summary.append({
                    'block': block_name,
                    'method': config.method,
                    'bits': config.bits,
                    'group_size': config.group_size,
                    'error': config.error_metric,
                    'memory_savings': config.memory_savings
                })
            else:
                print(f"  ⚠ {block_name}: Quantized block not found, applying config...")
                self.apply_quantization_config(block_name, config)
        
        print(f"\n📊 Quantization Summary:")
        print(f"  Blocks quantized: {len(quantization_summary)}")
        print(f"  Total estimated memory savings: {total_memory_savings:.1f}%")
        
        # Print breakdown by method
        method_counts = {}
        for summary in quantization_summary:
            method = summary['method']
            method_counts[method] = method_counts.get(method, 0) + 1
        
        for method, count in method_counts.items():
            print(f"  {method}: {count} blocks")
            
        return quantization_summary
        
    def apply_quantization_config(self, block_name: str, config: QuantizationConfig):
        """Apply the quantization configuration to the actual model block."""
        from .utils import get_module_by_name, set_module_by_name
        from .base_wrapper import RTNWrapper
        
        print(f"  Applying {config.method} quantization: {config.bits}-bit, group_size={config.group_size}")
        
        try:
            # Get the original block
            original_block = get_module_by_name(self.model, block_name)
            
            if config.method == "RTN":
                # Create RTN wrapper
                quantized_wrapper = RTNWrapper(
                    wrapped_block=original_block,
                    bits=config.bits,
                    group_size=config.group_size,
                    storage_device="cpu",
                    block_name=block_name
                )
                quantized_wrapper.enable()
                
                # Replace in model
                set_module_by_name(self.model, block_name, quantized_wrapper)
                
                # Update our tracking
                self.quantized_blocks[block_name] = quantized_wrapper
                self.current_model_state[block_name] = 'quantized'
                
            elif config.method in ["Trained", "Mixed"]:
                # TODO: Apply trained quantization
                print(f"    ⚠ {config.method} quantization application not yet implemented")
                
        except Exception as e:
            print(f"    ✗ Failed to apply {config.method} quantization to {block_name}: {e}")
            
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

    def finalize(self) -> nn.Module:
        """
        Finalize quantization by applying all successfully quantized blocks to the model.
        Returns the model with quantized blocks applied.
        """
        print("🔧 Finalizing quantization - applying saved quantized blocks...")
        
        if hasattr(self, 'saved_quantized_blocks') and self.saved_quantized_blocks:
            for block_name, quantized_block in self.saved_quantized_blocks.items():
                print(f"  Applying quantized block: {block_name}")
                self._replace_block_in_model(block_name, quantized_block)
                
            print(f"✅ Applied {len(self.saved_quantized_blocks)} quantized blocks to model")
        else:
            print("⚠️  No quantized blocks found - returning original model")
            
        return self.model
        
    def _replace_block_in_model(self, block_name: str, quantized_block):
        """Replace a block in the model with its quantized version."""
        parts = block_name.split('.')
        current = self.model
        
        # Navigate to parent of target block
        for part in parts[:-1]:
            current = getattr(current, part)
            
        # Replace the final block
        setattr(current, parts[-1], quantized_block)


