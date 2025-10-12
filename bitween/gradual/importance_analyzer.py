"""
Block and layer importance discovery through multiple analysis methods.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch.nn.functional as F


@dataclass
class ImportanceScore:
    """Container for importance analysis results."""
    block_name: str
    noise_sensitivity_ppl: float
    noise_sensitivity_kl: float
    rtn_8bit_impact: float
    rtn_4bit_impact: float
    rtn_2bit_impact: float


class ImportanceAnalyzer:
    """
    Discovers block and layer importance through multiple complementary methods.
    
    Methods implemented:
    1. Random noise injection testing
    2. RTN quantization impact testing (8/4/2 bit)  
    3. Block disable/exclusion testing
    4. Layer upgrade benefit analysis
    """
    
    def __init__(self, model: nn.Module, tokenizer, calibration_samples: int = 100, evaluation_samples: int = 2):
        self.model = model
        self.tokenizer = tokenizer
        self.calibration_samples = calibration_samples
        self.evaluation_samples = evaluation_samples
        
        # Results storage
        self.block_importance_scores = {}
        self.layer_importance_scores = {}
        self.analysis_cache = {}
        
    def analyze_block_importance(self, block_names: List[str] = None) -> Dict[str, ImportanceScore]:
        """
        Run all block importance discovery methods and aggregate results.
        
        Args:
            block_names: List of block names to analyze (avoids duplicate detection)
        
        Returns:
            Dictionary mapping block names to ImportanceScore objects
        """
        # Use provided block names or detect them
        self.block_names = block_names
        
        # Run noise injection test first
        noise_scores = self._run_noise_injection_test()
        
        # Run RTN sensitivity test
        # rtn_scores = self._run_rtn_sensitivity_test()
        rtn_scores = {'model.decoder.layers.0': {'8bit': 0.09429931640625, '4bit': 8.807220458984375, '2bit': 4067.5029296875}, 'model.decoder.layers.1': {'8bit': -0.1089324951171875, '4bit': 0.1566314697265625, '2bit': 3870.28955078125}, 'model.decoder.layers.10': {'8bit': -0.1432037353515625, '4bit': 4.478462219238281, '2bit': 1138.7763671875}, 'model.decoder.layers.11': {'8bit': -0.35028839111328125, '4bit': 10.711296081542969, '2bit': 1059.99365234375}, 'model.decoder.layers.2': {'8bit': -0.019073486328125, '4bit': 2.9447021484375, '2bit': 1762.01220703125}, 'model.decoder.layers.3': {'8bit': -0.0877685546875, '4bit': 3.07257080078125, '2bit': 1402.13427734375}, 'model.decoder.layers.4': {'8bit': -0.13086700439453125, '4bit': -1.4408645629882812, '2bit': 1261.44384765625}, 'model.decoder.layers.5': {'8bit': -0.0730743408203125, '4bit': 1.1270370483398438, '2bit': 1653.35400390625}, 'model.decoder.layers.6': {'8bit': 0.199615478515625, '4bit': 3.894287109375, '2bit': 3093.42724609375}, 'model.decoder.layers.7': {'8bit': -0.169036865234375, '4bit': 7.075172424316406, '2bit': 1839.68701171875}, 'model.decoder.layers.8': {'8bit': -0.24076080322265625, '4bit': 1.54425048828125, '2bit': 1324.6083984375}, 'model.decoder.layers.9': {'8bit': 0.0688323974609375, '4bit': 1.9608383178710938, '2bit': 1722.9541015625}}
        print(rtn_scores)
        
        # Combine noise injection and RTN scores
        importance_scores = {}
        for block_name in self.block_names:
            block_noise_scores = noise_scores.get(block_name, {'ppl_recovery': 0.0, 'kl_recovery': 0.0})
            block_rtn_scores = rtn_scores.get(block_name, {'8bit': 0.0, '4bit': 0.0, '2bit': 0.0})
            
            # Extract scores
            ppl_score = block_noise_scores['ppl_recovery']
            kl_score = block_noise_scores['kl_recovery']
            
            importance_scores[block_name] = ImportanceScore(
                block_name=block_name,
                noise_sensitivity_ppl=ppl_score,
                noise_sensitivity_kl=kl_score,
                rtn_8bit_impact=block_rtn_scores['8bit'],
                rtn_4bit_impact=block_rtn_scores['4bit'],  
                rtn_2bit_impact=block_rtn_scores['2bit']
            )
        
        return importance_scores
        
    def _run_noise_injection_test(self) -> Dict[str, Dict[str, float]]:
        """
        Run noise injection test to measure block importance.
        
        Returns:
            Dictionary mapping block names to dict with separate PPL and KL scores
        """
        # Get fixed calibration data
        fixed_input_tokens = self._get_fixed_calibration_data()
        base_seed = 42
        
        # Phase 1: Wrap all blocks with noise
        block_wrappers = {}
        for i, block_name in enumerate(self.block_names):
            block_wrappers[block_name] = self._wrap_block_with_noise(
                block_name, 
                seed=base_seed + i
            )
            
        # Phase 2: All blocks noisy baseline
        for wrapper in block_wrappers.values():
            wrapper.enable()
            
        degraded_baseline_ppl = self._evaluate_with_fixed_input(fixed_input_tokens)
        degraded_baseline_logits = self._calculate_kl_between_states(fixed_input_tokens, "all_noisy", "baseline")
        
        # Phase 3: Sequential recovery test
        recovery_scores = {}
        
        for i, target_block in enumerate(self.block_names):
            # Disable noise for this block only
            block_wrappers[target_block].disable()
            
            # Measure recovery
            recovered_ppl = self._evaluate_with_fixed_input(fixed_input_tokens)
            recovered_logits = self._calculate_kl_between_states(fixed_input_tokens, "target_recovered", target_block)
            
            # Calculate KL divergence between degraded and recovered states
            kl_recovery = self._compare_logits_for_kl(degraded_baseline_logits, recovered_logits)
            
            # Calculate recovery scores
            ppl_recovery = degraded_baseline_ppl - recovered_ppl

            # Store both scores separately for later analysis
            recovery_scores[target_block] = {
                'ppl_recovery': ppl_recovery,
                'kl_recovery': kl_recovery
            }
            
            # Re-enable noise for next iteration
            block_wrappers[target_block].enable()
            
        # Phase 4: Cleanup - remove all wrappers
        self._remove_all_wrappers(block_wrappers)
        
        return recovery_scores
        
    def _run_rtn_sensitivity_test(self) -> Dict[str, Dict[str, float]]:
        """
        Run RTN sensitivity test to measure block importance at different bit precisions.
        
        Tests 8-bit, 4-bit, and 2-bit quantization impact, then measures recovery
        when each block is restored to full precision.
        
        Returns:
            Dictionary mapping block names to dict with recovery scores for each precision
        """
        print("Running RTN sensitivity test...")
        
        # Get fixed calibration data (same as noise test)
        fixed_input_tokens = self._get_fixed_calibration_data()
        
        rtn_results = {}
        
        # Test each precision level
        for bits in [8, 4, 2]:
            
            # Phase 1: Wrap all blocks with RTN quantization
            block_wrappers = {}
            for block_name in self.block_names:
                block_wrappers[block_name] = self._wrap_block_with_rtn(
                    block_name, 
                    bits=bits,
                    group_size=32  # Standard group size
                )
                
            # Phase 2: All blocks quantized baseline
            for wrapper in block_wrappers.values():
                wrapper.enable()
                
            degraded_baseline_ppl = self._evaluate_with_fixed_input(fixed_input_tokens)
            degraded_baseline_logits = self._calculate_kl_between_states(fixed_input_tokens, f"all_{bits}bit", "baseline")
            
            
            # Phase 3: Sequential recovery test
            for target_block in self.block_names:
                
                # Restore this block to full precision
                block_wrappers[target_block].disable()
                
                # Measure recovery
                recovered_ppl = self._evaluate_with_fixed_input(fixed_input_tokens)
                recovered_logits = self._calculate_kl_between_states(fixed_input_tokens, f"target_recovered_{bits}bit", target_block)
                
                # Calculate KL divergence between quantized and recovered states
                kl_recovery = self._compare_logits_for_kl(degraded_baseline_logits, recovered_logits)
                
                # Calculate recovery scores
                ppl_recovery = degraded_baseline_ppl - recovered_ppl
                
                
                # Store results
                if target_block not in rtn_results:
                    rtn_results[target_block] = {}
                    
                rtn_results[target_block][f'{bits}bit'] = {
                    'ppl_recovery': ppl_recovery,
                    'kl_recovery': kl_recovery
                }
                
                # Re-enable quantization for next iteration
                block_wrappers[target_block].enable()
                
            # Phase 4: Cleanup - remove all wrappers for this precision
            self._remove_all_rtn_wrappers(block_wrappers)
            
        # Convert to expected format for ImportanceScore
        final_results = {}
        for block_name in self.block_names:
            final_results[block_name] = {
                '8bit': rtn_results.get(block_name, {}).get('8bit', {}).get('ppl_recovery', 0.0),
                '4bit': rtn_results.get(block_name, {}).get('4bit', {}).get('ppl_recovery', 0.0),
                '2bit': rtn_results.get(block_name, {}).get('2bit', {}).get('ppl_recovery', 0.0)
            }
            
        print("RTN sensitivity test completed")
        return final_results
        
    def _wrap_block_with_rtn(self, block_name: str, bits: int, group_size: int = 32):
        """Wrap a block with RTN quantization wrapper."""
        from .base_wrapper import RTNWrapper
        from .utils import get_module_by_name, set_module_by_name
        
        # Get the original block
        original_block = get_module_by_name(self.model, block_name)
        
        # Create RTN wrapper
        wrapper = RTNWrapper(
            wrapped_block=original_block,
            bits=bits,
            group_size=group_size,
            storage_device="cuda",
            block_name=block_name
        )
        
        # Replace the block with wrapper
        set_module_by_name(self.model, block_name, wrapper)
        
        return wrapper
        
    def _remove_all_rtn_wrappers(self, block_wrappers: Dict):
        """Remove all RTN wrappers and restore original blocks."""
        from .utils import set_module_by_name
        
        for block_name, wrapper in block_wrappers.items():
            # Clean up quantization first
            wrapper.cleanup()
            
            # Get original block from wrapper
            original_block = wrapper.get_wrapped_block()
            
            # Restore original block
            set_module_by_name(self.model, block_name, original_block)

    def _get_fixed_calibration_data(self):
        """Get fixed calibration data for consistent testing."""
        from ..calib_dataset import get_calibration_dataset
        
        # Get small, fixed dataset for testing
        calib_dataset = get_calibration_dataset(
            dataset_name="pile-10k",
            tokenizer=self.tokenizer,
            seqlen=512,
            nsamples=self.evaluation_samples,
            seed=123
        )
        
        # Get the tokenized samples - use evaluation_samples for evaluation (should be small like 2-5)
        samples = calib_dataset.get_samples(num_samples=self.evaluation_samples)
        return samples
        
    def _wrap_block_with_noise(self, block_name: str, seed: int):
        """Wrap a block with noise wrapper."""
        from .base_wrapper import NoiseWrapper
        from .utils import get_module_by_name, set_module_by_name
        
        # Get the original block
        original_block = get_module_by_name(self.model, block_name)
        
        # Create noise wrapper
        wrapper = NoiseWrapper(
            wrapped_block=original_block,
            noise_std=0.1,  # 10% noise relative to weight std
            seed=seed,
            storage_device="cuda",
            block_name=block_name
        )
        
        # Replace the block with wrapper
        set_module_by_name(self.model, block_name, wrapper)
        
        return wrapper
        
    def _remove_all_wrappers(self, block_wrappers: Dict):
        """Remove all wrappers and restore original blocks."""
        from .utils import set_module_by_name
        
        for block_name, wrapper in block_wrappers.items():
            # Get original block from wrapper
            original_block = wrapper.get_wrapped_block()
            
            # Restore original block
            set_module_by_name(self.model, block_name, original_block)
            
    def _evaluate_with_fixed_input(self, fixed_input_tokens) -> float:
        """Evaluate perplexity with fixed input tokens."""
        self.model.eval()
        device = next(self.model.parameters()).device
        
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for sample in fixed_input_tokens:
                input_ids = sample["input_ids"].unsqueeze(0).to(device)
                
                # Forward pass
                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs.loss
                
                total_loss += loss.item() * input_ids.size(1)
                total_tokens += input_ids.size(1)
                
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return perplexity
        
    def _calculate_kl_with_fixed_input(self, fixed_input_tokens) -> float:
        """
        Calculate KL divergence with fixed input tokens.
        Since we can't compare against original model (it's wrapped), 
        this returns a placeholder for now.
        """
        # TODO: Could store original model copy if needed for true KL divergence
        return 0.001  # Placeholder
        
    def _calculate_kl_between_states(self, fixed_input_tokens, baseline_state: str, recovered_state: str) -> float:
        """
        Calculate KL divergence between two model states for noise injection test.
        
        Args:
            fixed_input_tokens: Fixed calibration samples
            baseline_state: Description of baseline state (for logging)
            recovered_state: Description of recovered state (for logging)
            
        Returns:
            KL divergence between the two states
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        
        total_kl = 0.0
        total_tokens = 0
        
        # Get logits for current state (this should be called at the right time)
        current_logits = []
        
        with torch.no_grad():
            for sample in fixed_input_tokens:
                input_ids = sample["input_ids"].unsqueeze(0).to(device)
                
                # Get current state logits
                outputs = self.model(input_ids)
                logits = outputs.logits  # shape: [1, seq_len, vocab]
                current_logits.append(logits)
                
        return current_logits  # Return logits for comparison later
        
    def _compare_logits_for_kl(self, logits1_list, logits2_list) -> float:
        """
        Compare two sets of logits and calculate KL divergence.
        
        Args:
            logits1_list: List of logit tensors from first state
            logits2_list: List of logit tensors from second state
            
        Returns:
            Average KL divergence per token
        """
        total_kl = 0.0
        total_tokens = 0
        
        for logits1, logits2 in zip(logits1_list, logits2_list):
            # Ensure same length
            min_len = min(logits1.shape[1], logits2.shape[1])
            logits1 = logits1[:, :min_len, :]
            logits2 = logits2[:, :min_len, :]
            
            # Calculate KL divergence
            log_prob1 = F.log_softmax(logits1, dim=-1)
            prob2 = F.softmax(logits2, dim=-1)
            
            # Per-token KL divergence
            per_token_kl = F.kl_div(log_prob1, prob2, reduction='none', log_target=False)  
            per_token_kl = per_token_kl.sum(dim=-1)  # sum over vocab dim
            
            total_kl += per_token_kl.sum().item()
            total_tokens += per_token_kl.numel()
            
        avg_kl_per_token = total_kl / total_tokens if total_tokens > 0 else 0.0
        return avg_kl_per_token
        
    def calculate_block_budget_allocation(
        self,
        importance_scores: Dict[str, ImportanceScore],
        max_ppl_increase: float,
        max_kl_increase: float,
        ppl_weight: float = 0.7,
        kl_weight: float = 0.3,
        safety_multiplier: float = 2.0
    ) -> Dict[str, Dict]:
        """
        Convert raw importance scores to budget allocation percentages.
        
        Args:
            importance_scores: Raw importance scores from analyze_block_importance
            max_ppl_increase: Maximum allowed perplexity increase (e.g., 5.0 for 5%)
            max_kl_increase: Maximum allowed KL divergence increase (e.g., 0.01)
            ppl_weight: Weight for perplexity in combined score (0.0-1.0)
            kl_weight: Weight for KL divergence in combined score (0.0-1.0)
            safety_multiplier: Safety margin multiplier (default 2.0)
            
        Returns:
            Dictionary mapping block names to budget allocation info
        """
        # Calculate working budgets with safety margin
        working_ppl_budget = max_ppl_increase * safety_multiplier
        working_kl_budget = max_kl_increase * safety_multiplier
        
        # Step 1: Normalize scores by precision level
        normalized_scores = self._normalize_importance_scores(importance_scores)
        
        # Step 2: Calculate combined sensitivity scores
        combined_scores = {}
        for block_name, norm_scores in normalized_scores.items():
            # Combine PPL and KL sensitivities with weights
            ppl_sensitivity = (
                norm_scores['noise_ppl_norm'] * 0.3 +  # Noise injection weight
                norm_scores['rtn_4bit_norm'] * 0.7     # RTN 4-bit weight (most relevant)
            )
            
            kl_sensitivity = (
                norm_scores['noise_kl_norm'] * 0.4 +   # Noise injection weight
                norm_scores['rtn_4bit_norm'] * 0.6     # RTN sensitivity affects KL too
            )
            
            # Combined weighted sensitivity score
            combined_sensitivity = (ppl_weight * ppl_sensitivity + kl_weight * kl_sensitivity)
            
            combined_scores[block_name] = {
                'ppl_sensitivity': ppl_sensitivity,
                'kl_sensitivity': kl_sensitivity,
                'combined_sensitivity': combined_sensitivity
            }
        
        # Step 3: Calculate total sensitivity for normalization
        total_combined_sensitivity = sum(scores['combined_sensitivity'] for scores in combined_scores.values())
        
        # Step 4: Allocate budgets proportionally
        budget_allocations = {}
        
        print("Block Budget Allocations:")
        print("=" * 80)
        
        for block_name, scores in combined_scores.items():
            # Calculate percentage of total budget this block gets
            budget_percentage = scores['combined_sensitivity'] / total_combined_sensitivity
            
            # Allocate actual budget amounts
            allocated_ppl_budget = working_ppl_budget * budget_percentage
            allocated_kl_budget = working_kl_budget * budget_percentage
            
            budget_allocations[block_name] = {
                'ppl_budget_percent': budget_percentage * 100,
                'kl_budget_percent': budget_percentage * 100,
                'allocated_ppl_budget': allocated_ppl_budget,
                'allocated_kl_budget': allocated_kl_budget,
                'ppl_sensitivity': scores['ppl_sensitivity'],
                'kl_sensitivity': scores['kl_sensitivity'],
                'combined_sensitivity': scores['combined_sensitivity']
            }
            
        return budget_allocations
    
    def _normalize_importance_scores(self, importance_scores: Dict[str, ImportanceScore]) -> Dict[str, Dict]:
        """
        Normalize importance scores to 0-1 range for fair comparison.
        """
        # Collect all scores by metric
        all_noise_ppl = [score.noise_sensitivity_ppl for score in importance_scores.values()]
        all_noise_kl = [score.noise_sensitivity_kl for score in importance_scores.values()]
        all_rtn_8bit = [score.rtn_8bit_impact for score in importance_scores.values()]
        all_rtn_4bit = [score.rtn_4bit_impact for score in importance_scores.values()]
        all_rtn_2bit = [score.rtn_2bit_impact for score in importance_scores.values()]
        
        # Calculate normalization ranges (min-max normalization)
        def safe_normalize(values, scores):
            """Safely normalize scores, handling edge cases"""
            values = [max(0, v) for v in values]  # Ensure non-negative
            if max(values) == min(values):
                return {block: 0.5 for block in scores.keys()}  # Equal if all same
            
            value_range = max(values) - min(values)
            return {
                block: (max(0, score) - min(values)) / value_range
                for block, score in zip(scores.keys(), values)
            }
        
        # Normalize each metric
        noise_ppl_norm = safe_normalize(all_noise_ppl, importance_scores)
        noise_kl_norm = safe_normalize(all_noise_kl, importance_scores)
        rtn_8bit_norm = safe_normalize(all_rtn_8bit, importance_scores)
        rtn_4bit_norm = safe_normalize(all_rtn_4bit, importance_scores)
        rtn_2bit_norm = safe_normalize(all_rtn_2bit, importance_scores)
        
        # Combine into normalized score dictionary
        normalized_scores = {}
        for block_name in importance_scores.keys():
            normalized_scores[block_name] = {
                'noise_ppl_norm': noise_ppl_norm[block_name],
                'noise_kl_norm': noise_kl_norm[block_name],
                'rtn_8bit_norm': rtn_8bit_norm[block_name],
                'rtn_4bit_norm': rtn_4bit_norm[block_name],
                'rtn_2bit_norm': rtn_2bit_norm[block_name]
            }
        
        return normalized_scores
        
    def calculate_combined_sensitivity(
        self, 
        ppl_increase: float, 
        kl_increase: float,
        max_ppl_increase: float,
        max_kl_increase: float,
        ppl_weight: float = 0.7,
        kl_weight: float = 0.3
    ) -> float:
        """
        Calculate combined sensitivity score from PPL and KL increases.
        
        Args:
            ppl_increase: Perplexity increase from baseline
            kl_increase: KL divergence increase from baseline
            max_ppl_increase: Maximum expected perplexity increase for normalization
            max_kl_increase: Maximum expected KL increase for normalization
            ppl_weight: Weight for perplexity in combined score
            kl_weight: Weight for KL divergence in combined score
            
        Returns:
            Combined sensitivity score
        """
        # Normalize increases to 0-1 range for comparison
        ppl_normalized = min(ppl_increase / max_ppl_increase, 1.0) if max_ppl_increase > 0 else 0.0
        kl_normalized = min(kl_increase / max_kl_increase, 1.0) if max_kl_increase > 0 else 0.0
        
        # Calculate weighted combined sensitivity
        combined_sensitivity = (ppl_weight * ppl_normalized + kl_weight * kl_normalized)
        
        return combined_sensitivity