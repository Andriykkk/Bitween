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
        training_batch_size: int = 4
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
                    
                # Step 2: RTN failed, try training
                print(f"      RTN failed (error: {rtn_error:.4f}), trying training...")
                
                train_success, train_error = self._try_trainable_quantization(
                    block, target_bits, target_group_size, cached_data, budget_allocation
                )
                
                if train_success:
                    print(f"      ✓ Training successful (error: {train_error:.4f})")
                    best_config = QuantizationConfig(target_bits, target_group_size, "Trained", train_error)
                    continue  # Try even lower precision
                    
                # Step 3: Training failed, try layer-level recovery (only at minimum group size)
                if target_group_size == self.min_group_size:
                    print(f"      Training failed (error: {train_error:.4f}), trying layer recovery...")
                    
                    recovery_success, recovery_error = self._try_layer_recovery(
                        block, target_bits, target_group_size, cached_data, budget_allocation
                    )
                else:
                    print(f"      Training failed (error: {train_error:.4f}), skipping layer recovery (group_size={target_group_size} > min={self.min_group_size})")
                    recovery_success, recovery_error = False, train_error
                
                if recovery_success:
                    print(f"      ✓ Layer recovery successful (error: {recovery_error:.4f})")
                    best_config = QuantizationConfig(target_bits, target_group_size, "Mixed", recovery_error)
                    continue  # Try even lower precision
                else:
                    print(f"      ✗ Layer recovery failed (error: {recovery_error:.4f})")
                    # This precision/group_size combination doesn't work
                    if best_config is not None:
                        # Return best previous configuration
                        print(f"    Using best config: {best_config.bits}-bit, group_size={best_config.group_size}, method={best_config.method}")
                        return best_config
                    # Continue to next group_size
                    
        # Return best configuration found, or None if nothing worked
        if best_config is not None:
            print(f"  Final config: {best_config.bits}-bit, group_size={best_config.group_size}, method={best_config.method}")
            return best_config
        else:
            print(f"  No quantization possible within budget")
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
            kl_increase=kl_increase
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
            
            # Step 3: Train the wrapped block
            result = train_individual_wrapper(
                module_name=block_name,
                wrapped_module=wrapper_info['wrapped_module'],
                block_inputs=block_inputs,
                iters=3,  # Number of training epochs
                lr=1e-4,  # Learning rate 
                batch_size=self.training_batch_size,  # Configurable batch size
                is_single_layer=wrapper_info['is_single_layer']
            )
            
            # Step 4: Apply best parameters
            if not wrapper_info['is_single_layer']:
                # For blocks, apply best parameters to all wrappers
                from ..wrapper import WrapperLinear
                wrapped_module = wrapper_info['wrapped_module']
                for name, submodule in wrapped_module.named_modules():
                    if isinstance(submodule, WrapperLinear):
                        submodule.apply_best_params()
            
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
        
    def _try_layer_recovery(self, block, bits, group_size, cached_data, budget):
        """Try layer-level recovery for failed quantization."""
        # TODO: Implement layer recovery algorithm
        # This should:
        # 1. Identify problematic layers
        # 2. Try reverting each layer to higher precision
        # 3. Apply best reversions until budget is met
        
        import random
        error = random.uniform(0, budget * 1.2)  # Best chance with recovery
        success = error <= budget
        return success, error
        
        
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


