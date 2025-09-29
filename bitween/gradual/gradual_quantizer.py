"""
Main orchestrator for gradual/adaptive quantization process.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import copy

from .importance_analyzer import ImportanceAnalyzer
from .precision_optimizer import PrecisionOptimizer, OptimizationResult
from .evaluator import GradualEvaluator
from .utils import get_transformer_block_names
from ..utils.evaluation import calculate_perplexity



class GradualQuantizer:
    """
    Main controller for gradual quantization process.
    
    Orchestrates the complete pipeline:
    1. Block/layer importance discovery
    2. Progressive precision reduction (8bit -> 4bit -> 2bit)
    3. Dynamic threshold management
    4. Global optimization and validation
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        max_perplexity_increase: float = 5,
        max_per_token_kl_divergence: float = 0.01,
        safety_multiplier: float = 2.0,
        nsamples: int = 128,
        evaluation_samples: int = 5,
        ignore_layers: Optional[List[str]] = None,
        cpu_offload: bool = False
    ):
        """
        Initialize gradual quantizer.
        
        Args:
            model: The model to quantize
            tokenizer: Tokenizer for the model
            max_perplexity_increase: Maximum allowed perplexity increase in percent (e.g., 5 for 5%)
            max_per_token_kl_divergence: Maximum allowed KL divergence per token
            safety_multiplier: Multiply budget by this for working room
            calibration_samples: Samples for importance analysis
            evaluation_samples: Samples for final evaluation (kept small for speed)
            ignore_layers: List of layer names to exclude from quantization
            cpu_offload: Enable CPU offloading for memory efficiency (requires GPU)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_perplexity_increase = max_perplexity_increase
        self.max_per_token_kl_divergence = max_per_token_kl_divergence
        self.safety_multiplier = safety_multiplier
        self.calibration_samples = nsamples
        self.evaluation_samples = evaluation_samples

        self.ignore_layers = ignore_layers
        
        # Device management - detect model's current device
        self.gpu_available = torch.cuda.is_available()
        self.model_device = next(self.model.parameters()).device
        self.cpu_offload = cpu_offload if self.gpu_available and self.model_device.type == 'cuda' else False
        
        # Determine storage and working devices
        if self.cpu_offload:
            self.wrapper_storage_device = "cpu"
            self.working_device = "cuda"
            print("💾 CPU offload enabled - blocks will be stored on CPU and loaded to GPU for forward pass")
        else:
            self.wrapper_storage_device = str(self.model_device)
            self.working_device = str(self.model_device)
            if not self.gpu_available:
                print("⚠️  No GPU available - running on CPU only")
        
        # Working budget with safety margin
        self.working_budget = max_perplexity_increase * safety_multiplier
        
        # Initialize components with device settings
        self.importance_analyzer = ImportanceAnalyzer(
            model, tokenizer, 
            calibration_samples=self.calibration_samples,  # For training (128)
            evaluation_samples=self.evaluation_samples,  # For evaluation (5)
            wrapper_storage_device=self.wrapper_storage_device,
            working_device=self.working_device
        )
        self.precision_optimizer = PrecisionOptimizer(model, tokenizer)
        self.evaluator = GradualEvaluator(model, tokenizer)
        
        # State tracking
        self.quantization_state = None
        self.importance_scores = None
        self.baseline_metrics = None
        
    def quantize(self) -> nn.Module:
        """
        Main entry point for gradual quantization.
        
        Focus is on minimizing memory by reducing precision while staying within
        perplexity and KL divergence constraints.
        
        Returns:
            Quantized model optimized for maximum precision reduction within quality limits
        """
        # Phase 1: Baseline establishment and importance discovery
        self._establish_baseline()
        self._discover_importance()

                # Print importance scores for each block
        # print("\n" + "="*60)
        # print("BLOCK IMPORTANCE SCORES")
        # print("="*60)
        # for block_name, score in self.importance_scores.items():
        #     print(f"{block_name}:")
        #     print(f"  Noise Sensitivity (PPL):    {score.noise_sensitivity_ppl:.6f}")
        #     print(f"  Noise Sensitivity (KL):     {score.noise_sensitivity_kl:.6f}")
        #     print(f"  RTN 8-bit Impact:           {score.rtn_8bit_impact:.6f}")
        #     print(f"  RTN 4-bit Impact:           {score.rtn_4bit_impact:.6f}")
        #     print(f"  RTN 2-bit Impact:           {score.rtn_2bit_impact:.6f}")
        #     print()
        # print("="*60)
        
        # Phase 2: Progressive quantization with budget management
        self._progressive_quantization()
        
        # Phase 3: Global optimization and validation
        self._global_optimization()
        
        # Phase 4: Final evaluation and cleanup
        quantized_model = self._finalize_quantization()
        
        return quantized_model
        
    def _establish_baseline(self):
        """Measure original model performance including perplexity and KL divergence baseline."""
        # Calculate baseline perplexity
        baseline_ppl = calculate_perplexity(
            model=self.model,
            tokenizer=self.tokenizer,
            eval_samples=self.evaluation_samples,
            verbose=False
        )
        
        self.baseline_metrics = {
            'perplexity': baseline_ppl,
            'perplexity_budget_absolute': baseline_ppl * (self.max_perplexity_increase / 100.0),
            'working_budget_absolute': baseline_ppl * (self.working_budget / 100.0),
            'kl_token_budget': self.max_per_token_kl_divergence
        }
        
        return self.baseline_metrics
        
    def _discover_importance(self):
        """Run all importance discovery methods and aggregate results."""
        block_names = get_transformer_block_names(self.model, ignore_layers=self.ignore_layers)

        print(f"Discovering importance for {len(block_names)} blocks: {block_names}")
        
        # Pass block names to analyzer to avoid duplicate detection
        self.importance_scores = self.importance_analyzer.analyze_block_importance(block_names)
        
        return self.importance_scores
        
    def _progressive_quantization(self):
        """Apply quantization progressively based on importance scores."""
        if not self.importance_scores:
            raise ValueError("Must run importance analysis first")
        
        # Calculate budget allocation based on importance scores
        budget_allocations = self.importance_analyzer.calculate_block_budget_allocation(
            importance_scores=self.importance_scores,
            max_ppl_increase=self.max_perplexity_increase,
            max_kl_increase=self.max_per_token_kl_divergence,
            ppl_weight=0.7,  # PPL is more important than KL
            kl_weight=0.3,
            safety_multiplier=self.safety_multiplier
        )
        
        # Store budget allocations for later use
        self.budget_allocations = budget_allocations
        
        # Set up block training system for block-by-block training
        self._setup_block_training()
        
        # Process all blocks sequentially using training system
        self._process_blocks_with_training()
    
    def _quantize_block_progressive(self, block_name: str, allocation: Dict):
        """
        Quantize a block progressively: 8-bit → 4-bit → 2-bit with group size optimization.
        """
        allocated_budget = allocation['allocated_ppl_budget']
        
        print(f"  Budget: {allocated_budget:.3f} ({allocation['ppl_budget_percent']:.1f}%)")
        
        # Try progressive quantization: 8→4→2 with group size optimization
        for bits in [4]:  # Start with 4-bit for now
            print(f"    Trying {bits}-bit quantization...")
            
            # Try different group sizes (smaller = better quality)
            for group_size in [16, 32, 64, 128]:
                result = self.precision_optimizer.optimize_block(
                    block_name=block_name,
                    target_bits=bits,
                    budget_limit=allocated_budget
                )
                
                if result.result == OptimizationResult.SUCCESS:
                    print(f"    ✓ Success: {bits}-bit with group_size={group_size}")
                    break
                else:
                    print(f"    ✗ Failed: {bits}-bit with group_size={group_size} ({result.result})")
            
            # If this precision level worked, we're done
            if result.result == OptimizationResult.SUCCESS:
                break
                
        if result.result != OptimizationResult.SUCCESS:
            print(f"  ⚠️  Block couldn't be quantized within budget - needs layer analysis")
            
    def _quantize_block_layer_by_layer(self, block_name: str, allocation: Dict):
        """
        Quantize a sensitive block layer by layer with individual budgets.
        """
        print(f"  Implementing layer-by-layer quantization for {block_name}")
        # TODO: Implement detailed layer-level quantization
        # This would involve:
        # 1. Detect all linear layers in the block
        # 2. Analyze each layer's quantization sensitivity
        # 3. Allocate sub-budgets to each layer
        # 4. Quantize layers individually: some to 8-bit, others to 4-bit, etc.
        # 5. Optimize group sizes per layer
        pass
        
    def _setup_block_training(self):
        """
        Set up block training system for block-by-block training.
        """
        from .base_wrapper import BlockTrainingManager
        from ..calib_dataset import get_calibration_dataset
        
        # Initialize block training manager
        self.training_manager = BlockTrainingManager(self.model, self.tokenizer)
        
        # Get all block names from importance analysis
        block_names = list(self.importance_scores.keys())
        
        # Wrap blocks for training
        self.training_manager.wrap_blocks_for_training(block_names)
        
        # Prepare calibration dataset for training (use calibration_samples)
        print(f"Preparing calibration dataset with {self.calibration_samples} samples...")
        calib_dataset = get_calibration_dataset(
            dataset_name="pile-10k",
            tokenizer=self.tokenizer,
            seqlen=512,  # Standard sequence length
            nsamples=self.calibration_samples,  # Use calibration_samples for training data
            seed=42  # Fixed seed for reproducibility
        )
        
        # Get samples for training (all calibration samples for training)
        self.dataset_samples = calib_dataset.get_samples(num_samples=self.calibration_samples)
        
        # Debug: Check sample format
        if self.dataset_samples:
            sample = self.dataset_samples[0]
            print(f"Sample format check:")
            print(f"  Type: {type(sample)}")
            print(f"  Keys: {list(sample.keys()) if isinstance(sample, dict) else 'Not a dict'}")
            for key, value in sample.items():
                if torch.is_tensor(value):
                    print(f"  {key}: shape={value.shape}, device={value.device}, dtype={value.dtype}")
                else:
                    print(f"  {key}: {type(value)} = {value}")
        
        print(f"Prepared {len(self.dataset_samples)} samples for block training")
        
    def _process_blocks_with_training(self):
        """
        Process all blocks sequentially using the block training system.
        """
        print("Starting block-by-block processing with training...")
        
        block_count = 0
        for block_name, cached_data in self.training_manager.process_all_blocks_sequentially(
            self.dataset_samples, 
            max_samples=self.calibration_samples
        ):
            block_count += 1
            
            print(f"\n=== Processing Block {block_count}: {block_name} ===")
            print(f"Cached {len(cached_data)} samples for training")
            
            # Get budget allocation for this block
            allocation = self.budget_allocations.get(block_name, {})
            if allocation:
                budget = allocation.get('allocated_ppl_budget', 0)
                budget_percent = allocation.get('ppl_budget_percent', 0)
                print(f"Allocated budget: {budget:.3f} ({budget_percent:.1f}%)")
            else:
                print(f"Warning: No budget allocation found for {block_name}")
                
            # Train this block on cached data
            self._train_block_on_cached_data(block_name, cached_data, allocation)
            
            # Clear cache to free memory
            self.training_manager.clear_block_cache(block_name)
            
            print(f"Completed processing {block_name}")
            
        print(f"\nCompleted processing {block_count} blocks")
        
        # Cleanup training system
        self.training_manager.cleanup()
        
    def _train_block_on_cached_data(self, block_name: str, cached_data: List, allocation: Dict):
        """
        Train a specific block on its cached activation data.
        
        Args:
            block_name: Name of block to train
            cached_data: List of (input_dict, output_tensor) pairs
            allocation: Budget allocation for this block
        """
        if not cached_data:
            print(f"No cached data for {block_name}, skipping training")
            return
            
        print(f"Training {block_name} on {len(cached_data)} cached samples...")
        
        # TODO: Implement actual training logic
        # This should:
        # 1. Get budget from allocation
        # 2. Try progressive quantization (8->4->2 bit)
        # 3. Optimize group sizes (128->64->32->16)
        # 4. Use trainable quantization if RTN fails
        # 5. Rollback if budget exceeded
        
        # For now, just simulate training
        import time
        time.sleep(0.1)  # Simulate training time
        
        print(f"Training simulation completed for {block_name}")
        
    def _global_optimization(self):
        """Perform end-to-end optimization and handle budget violations."""
        pass
        
    def _finalize_quantization(self) -> nn.Module:
        """Convert to final quantized model and cleanup temporary state."""
        pass
        
    def get_quantization_report(self) -> Dict:
        """
        Generate comprehensive report of quantization process.
        
        Returns:
            Dictionary containing metrics, decisions, and performance data
        """
        pass
        
    def get_precision_analysis(self) -> Dict:
        """
        Analyze precision assignment across model blocks and layers.
        
        Returns:
            Dictionary with precision statistics and bit distribution
        """
        pass
        
    def validate_kl_constraints(self) -> Dict[str, bool]:
        """
        Validate that KL divergence constraints are met at block/layer level.
        
        Returns:
            Dictionary mapping block/layer names to constraint satisfaction
        """
        pass
        
    def save_quantization_state(self, path: str):
        """Save current quantization state for resuming or analysis.""" 
        pass
        
    def load_quantization_state(self, path: str):
        """Load previously saved quantization state."""
        pass


class QuantizationState:
    """
    Tracks the current precision configuration of all blocks and layers.
    """
    
    def __init__(self):
        self.block_precisions = {}  # block_name -> bit precision
        self.layer_precisions = {}  # layer_name -> bit precision  
        self.group_sizes = {}       # module_name -> group_size
        self.ignored_layers = set() # layers excluded from quantization
        self.precision_history = [] # track changes for rollback
        
    def set_block_precision(self, block_name: str, bits: int):
        """Set precision for entire block."""
        pass
        
    def set_layer_precision(self, layer_name: str, bits: int):
        """Set precision for specific layer."""
        pass
        
    def get_current_precision(self, module_name: str) -> int:
        """Get current precision for module."""
        pass
        
    def rollback_to_checkpoint(self, checkpoint_name: str):
        """Rollback to previously saved state."""
        pass
        
    def create_checkpoint(self, checkpoint_name: str):
        """Save current state as checkpoint.""" 
        pass
        
    def apply_to_model(self, model: nn.Module):
        """Apply current precision configuration to model."""
        pass
        
    def get_precision_summary(self) -> Dict:
        """
        Get summary of current precision configuration.
        
        Returns:
            Dictionary with precision distribution statistics
        """
        pass
        
    def get_average_precision(self) -> float:
        """Calculate average precision across all quantized modules."""
        pass
        
    def get_precision_distribution(self) -> Dict[int, int]:
        """Get count of modules at each precision level."""
        pass