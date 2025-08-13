import torch
import torch.nn as nn
import torch.nn.functional as F
from .functional import quantize_rtn

class QuantizedLinear(nn.Module):
    """
    A quantized linear layer module that supports both 4-bit and 8-bit weights.
    """
    def __init__(self, in_features, out_features, bits=8, group_size=-1, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        
        # Determine the shape of the quantized weight tensor
        q_in_features = in_features if bits == 8 else in_features // 2
        self.register_buffer('qweight', torch.zeros((out_features, q_in_features), dtype=torch.uint8))

        # Determine the shape of the scale and zero_point buffers
        if group_size == -1:
            scale_zp_shape = (out_features, 1)
        else:
            num_groups = in_features // group_size
            scale_zp_shape = (out_features, num_groups, 1)
            
        self.register_buffer('scale', torch.zeros(scale_zp_shape))
        self.register_buffer('zero_point', torch.zeros(scale_zp_shape))
        
        if bias:
            self.register_buffer('bias', torch.zeros(out_features))
        else:
            self.bias = None

    def _dequantize_8bit(self):
        """Dequantizes 8-bit weights."""
        if self.group_size == -1:
            # Per-row dequantization
            w = (self.qweight.float() - self.zero_point) * self.scale
        else:
            # Group-wise dequantization
            q_w_view = self.qweight.view(self.out_features, -1, self.group_size).float()
            w = (q_w_view - self.zero_point) * self.scale
            w = w.view(self.out_features, self.in_features)
        return w

    def _dequantize_4bit(self):
        """Dequantizes 4-bit weights by unpacking them first."""
        # Unpack two 4-bit values from each uint8 byte
        low_bits = self.qweight & 0x0F
        high_bits = (self.qweight & 0xF0) >> 4
        
        # Interleave the unpacked values to restore original order
        unpacked_qweight = torch.stack((low_bits, high_bits), dim=-1).view(self.out_features, -1)

        if self.group_size == -1:
            # Per-row dequantization
            w = (unpacked_qweight.float() - self.zero_point) * self.scale
        else:
            # Group-wise dequantization
            q_w_view = unpacked_qweight.view(self.out_features, -1, self.group_size).float()
            w = (q_w_view - self.zero_point) * self.scale
            w = w.view(self.out_features, self.in_features)
        return w

    def dequantize(self):
        """Dispatches to the correct dequantization function based on bit width."""
        if self.bits == 8:
            return self._dequantize_8bit()
        elif self.bits == 4:
            return self._dequantize_4bit()
        else:
            raise NotImplementedError(f"Dequantization for {self.bits}-bit is not supported.")

    def forward(self, x):
        w = self.dequantize()
        return F.linear(x, w, self.bias)

    @classmethod
    def from_float(cls, float_layer: nn.Linear, bits=8, group_size=-1):
        q_weight, scale, zp, group_size_ret = quantize_rtn(float_layer.weight.data, bits=bits, group_size=group_size)
        
        q_layer = cls(float_layer.in_features, float_layer.out_features, bits, group_size_ret, bias=float_layer.bias is not None)

        q_layer.qweight.copy_(q_weight)
        q_layer.scale.copy_(scale)
        q_layer.zero_point.copy_(zp)
        if float_layer.bias is not None:
            q_layer.bias.data.copy_(float_layer.bias.data)

        return q_layer

    def __repr__(self):
        return (f"QuantizedLinear(in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, bits={self.bits}, group_size={self.group_size})")