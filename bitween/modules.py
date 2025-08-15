import torch
import torch.nn as nn
import torch.nn.functional as F
from bitween.functional import quantize_rtn
from .kernels.int import quantized_linear_kernel
import triton

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
    
    def dequantize(self):
        """Dispatches to the correct dequantization function based on bit width."""
        if self.bits == 8:
            return self._dequantize_8bit()
        else:
            raise NotImplementedError(f"Dequantization for {self.bits}-bit is not supported.")

    def forward(self, x):
        device = x.device

        original_shape = x.shape
        x = x.reshape(-1, original_shape[-1])

        M, K = x.shape
        N, _ = self.qweight.shape

        # Allocate the output tensor
        c = torch.empty((M, N), device=device, dtype=torch.float32)

        # Block sizes and grid dimensions
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = 64
        grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

        # Launch the kernel
        quantized_linear_kernel[grid](
            x, self.qweight, self.scale, self.zero_point, self.bias, c,
            M, N, K,
            x.stride(0), x.stride(1),
            self.qweight.stride(0), self.qweight.stride(1),
            self.scale.stride(0),
            self.zero_point.stride(0),
            self.bias.stride(0) if self.bias is not None else 0,
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BIAS_ENABLED=(self.bias is not None)
        )
        
        return c.reshape(*original_shape[:-1], N)

    @classmethod
    def from_float(cls, float_layer: nn.Linear, bits=8, group_size=-1):
        qweight, scale, zero_point, _ = quantize_rtn(float_layer.weight.data, bits=8, group_size=-1)

        device = float_layer.weight.device

        qweight = qweight.contiguous()
        zero_point = zero_point.contiguous()
        group_size_ret = group_size

        qweight = qweight.to(device)
        scale = scale.to(device)
        zero_point = zero_point.to(device)

        q_layer = cls(float_layer.in_features, float_layer.out_features, bits, group_size_ret, bias=float_layer.bias is not None)

        q_layer.qweight.copy_(qweight)
        q_layer.scale.copy_(scale)
        q_layer.zero_point.copy_(zero_point)
        if float_layer.bias is not None:
            q_layer.bias.data.copy_(float_layer.bias.data)

        return q_layer

    def __repr__(self):
        return (f"QuantizedLinear(in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, bits={self.bits}, group_size={self.group_size})")