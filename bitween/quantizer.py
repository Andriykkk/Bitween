import torch
import torch.nn as nn
from .modules import QuantizedLinear
from .evaluation import calculate_perplexity, calculate_kl_divergence, print_report
import copy
import gc

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
    def __init__(self, model, tokenizer=None, bits=8, group_size=-1):
        self.model = model
        self.tokenizer = tokenizer
        self.bits = bits
        self.group_size = group_size
        print(f"Bitween initialized with bits={bits}, group_size={group_size}")

    def quantize(self, evaluate_perplexity=False, num_samples=100, print_paddings=False, **eval_kwargs):
        """
        Quantizes the linear layers of the model in-place.
        
        Args:
            evaluate_perplexity (bool): If True, run a full performance evaluation.
            num_samples (int): Number of samples to use for evaluation.
            print_paddings (bool): If True, print padding to weights.
            **eval_kwargs: Additional arguments for the evaluation functions 
                          (e.g.).
        """
        original_ppl = None
        if evaluate_perplexity and self.tokenizer is not None and num_samples > 0:
            if self.tokenizer is None:
                raise ValueError("Tokenizer must be provided for evaluation.")
            print("\n--- Evaluating original model ---")
            original_ppl = calculate_perplexity(self.model, self.tokenizer, num_samples=num_samples, **eval_kwargs)

        print("\n--- Starting quantization ---")
        
        # Create a deepcopy for quantization to keep the original model intact for KL-divergence
        quantized_model = copy.deepcopy(self.model)
        
        # Find and replace all linear layers
        for name, module in quantized_model.named_modules():
            if isinstance(module, nn.Linear):
                print(f"  - Quantizing layer: {name}")
                
                q_module = QuantizedLinear.from_float(
                    module, self.bits, self.group_size, print_paddings
                )
                
                _set_module(quantized_model, name, q_module)
                
        print("--- Quantization complete ---")

        if evaluate_perplexity and self.tokenizer is not None and num_samples > 0:
            print("\n--- Evaluating quantized model ---")
            quantized_ppl = calculate_perplexity(quantized_model, self.tokenizer, num_samples=num_samples, **eval_kwargs)
            
            print("\n--- Calculating KL-Divergence ---")
            kl_div = calculate_kl_divergence(self.model, quantized_model, self.tokenizer, num_samples=num_samples, **eval_kwargs)
            
            print_report(original_ppl, quantized_ppl, kl_div)

        return quantized_model

