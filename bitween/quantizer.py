import torch
import torch.nn as nn
from .modules import QuantizedLinear

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
    def __init__(self, model, bits=8, group_size=-1):
        self.model = model
        self.bits = bits
        self.group_size = group_size
        print(f"MyQuantizer initialized with bits={bits}, group_size={group_size}")

    def quantize(self):
        """
        Quantizes the linear layers of the model in-place.
        """
        print("Starting quantization...")
        
        # Find and replace all linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                print(f"  - Quantizing layer: {name}")
                
                # Create a quantized version of the linear layer
                q_module = QuantizedLinear.from_float(
                    module, self.bits, self.group_size
                )
                
                # Replace the original layer with the new quantized layer
                _set_module(self.model, name, q_module)
                
        print("Quantization complete.")
        return self.model
