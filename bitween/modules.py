import torch
import torch.nn as nn
import torch.nn.functional as F
from .functional import quantize_rtn

class QuantizedLinear(nn.Module):
    """
    A quantized linear layer module.
    
    This layer stores weights in a quantized format and de-quantizes them
    on-the-fly during the forward pass.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Register buffers for quantized weights, scales, and zero points
        self.register_buffer('qweight', torch.zeros((out_features, in_features), dtype=torch.uint8))
        self.register_buffer('scale', torch.zeros((out_features, 1)))
        self.register_buffer('zero_point', torch.zeros((out_features, 1)))
        
        if bias:
            self.register_buffer('bias', torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x):
        # --- De-quantize the weights ---
        # 1. Convert qweight to the same dtype as scale
        # 2. Subtract the zero point
        # 3. Multiply by the scale
        dequantized_weight = (self.qweight.to(self.scale.dtype) - self.zero_point) * self.scale
        
        # --- Perform the linear operation ---
        return F.linear(x, dequantized_weight, self.bias)

    @classmethod
    def from_float(cls, float_module, bits=4, group_size=-1):
        """
        Creates a QuantizedLinear layer from a standard nn.Linear layer.
        
        Args:
            float_module (nn.Linear): The original, full-precision linear layer.
            bits (int): The number of bits for quantization.
            group_size (int): The group size for quantization.
            
        Returns:
            QuantizedLinear: The new quantized linear layer.
        """
        # 1. Create a new instance of our quantized layer
        q_module = cls(float_module.in_features, float_module.out_features, float_module.bias is not None)
        
        # 2. Quantize the weights from the float module
        q_weight, scale, zero_point = quantize_rtn(
            float_module.weight.data, bits=bits, group_size=group_size
        )
        
        # 3. Assign the quantized data to the new module's buffers
        q_module.qweight.data.copy_(q_weight)
        q_module.scale.data.copy_(scale)
        q_module.zero_point.data.copy_(zero_point)
        if float_module.bias is not None:
            q_module.bias.data.copy_(float_module.bias.data)
            
        return q_module

    def __repr__(self):
        return (f"QuantizedLinear(in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, bits={self.bits}, group_size={self.group_size})")
