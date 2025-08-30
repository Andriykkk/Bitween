import torch
import torch.nn as nn
import torch.nn.functional as F
from bitween.functional import quantize_rtn
from .kernels.int import quantized_linear_kernel
import triton
import triton.language as tl
import copy

class QuantizedLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, qweight, scale, zero_point, bias, bits, group_size):
        # Save tensors for backward pass
        ctx.save_for_backward(x, qweight, scale, zero_point, bias)

        ctx.bits = bits
        ctx.group_size = group_size

        # The fast, non-differentiable forward pass using the Triton kernel
        device = x.device
        original_shape = x.shape
        
        # Use 16-bit dtype based on GPU capability
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            target_dtype = torch.bfloat16
        else:
            target_dtype = torch.float16
        
        # Convert input to target dtype before kernel
        x_reshaped = x.reshape(-1, original_shape[-1]).to(target_dtype)

        M, K = x_reshaped.shape
        N, _ = qweight.shape
        
        c = torch.empty((M, N), device=device, dtype=target_dtype)
        # c = torch.empty((M, N), device=device, dtype=torch.int32)

        DTYPE = {
            torch.float16: tl.float16,
            torch.bfloat16: tl.bfloat16,
            torch.float32: tl.float32
        }

        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(N, meta['BLOCK_SIZE_N']))

        quantized_linear_kernel[grid](
            x_reshaped, qweight, scale, zero_point, bias, c,
            M, N, K,
            bits,
            (1 << bits) - 1,
            x_reshaped.stride(0), x_reshaped.stride(1),
            qweight.stride(0), qweight.stride(1),
            scale.stride(0), scale.stride(1),    
            zero_point.stride(0), zero_point.stride(1),
            bias.stride(0) if bias is not None else 0,
            c.stride(0), c.stride(1),
            GROUP_SIZE=group_size,
            BIAS_ENABLED=(bias is not None),
            DTYPE=DTYPE[target_dtype]
        )
        
        return c.reshape(*original_shape[:-1], N)

    @staticmethod
    def backward(ctx, grad_output):
        x, qweight, scale, zero_point, bias = ctx.saved_tensors

        # Only compute grad_input (treat qweight as frozen)
        grad_input = None

        out_features, packed_cols = qweight.shape
        values_per_int32 = 32 // ctx.bits
        group_size = ctx.group_size
        in_features = (packed_cols * values_per_int32)
        num_groups = in_features // group_size

        # Unpack and dequantize weights
        total_vals = out_features * num_groups * group_size
        q_flat = torch.empty(total_vals, dtype=torch.int32, device=qweight.device)

        for i in range(values_per_int32):
            shift = i * ctx.bits
            mask = (1 << ctx.bits) - 1
            q_vals = (qweight.view(-1) >> shift) & mask
            q_flat[i::values_per_int32] = q_vals

        q = q_flat.view(out_features, num_groups, group_size)
        scale = scale.view(out_features, num_groups, 1)
        zero_point = zero_point.view(out_features, num_groups, 1)
        weight = scale * (q.float() - zero_point.float())
        weight = weight.view(out_features, in_features)

        # Gradient for input x
        if ctx.needs_input_grad[0]:
            grad_input = grad_output @ weight

        # Return None for all other non-differentiable inputs
        return grad_input, None, None, None, None, None, None

class QuantizedLinear(nn.Module):
    """
    A quantized linear layer module that supports both 4-bit and 8-bit weights.
    """
    def __init__(self, in_features, out_features, bits=8, group_size=32, bias=True, dtype=torch.bfloat16):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size

        self.requires_grad = False

        if self.bits not in [2, 4, 8]:
            raise ValueError("Only 2, 4, or 8 bits are supported.")
        
        if group_size < 1:
            raise ValueError("Group size must be at least 1.")

        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 and dtype == torch.bfloat16:
            # Ampere (SM 8.0+) and newer support bfloat16 natively
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float16
        
        # Determine the shape of the quantized weight tensor
        values_per_int32 = 32 // self.bits
        q_in_features = self.in_features // values_per_int32
        self.register_buffer('qweight', torch.zeros((self.out_features, q_in_features), dtype=torch.int32))

        num_groups = in_features // group_size
        scale_zp_shape = (out_features, num_groups)
            
        self.register_buffer('scale', torch.zeros(scale_zp_shape, dtype=self.dtype))
        self.register_buffer('zero_point', torch.zeros(scale_zp_shape, dtype=self.dtype))
        
        if bias:
            self.register_buffer('bias', torch.zeros(out_features, dtype=self.dtype))
        else:
            self.bias = None
 
    def forward(self, x):
        # if x.dtype != torch.bfloat16 or x.dtype != torch.float16:
        #     x = x.to(self.dtype)
        x = QuantizedLinearFunction.apply(x, self.qweight, self.scale, self.zero_point, self.bias, self.bits, self.group_size)
        return x

    @classmethod
    def from_float(cls, float_layer: nn.Linear, bits=8, group_size=32):
        float_layer = copy.deepcopy(float_layer).to(torch.float32)
        qweight, scale, zero_point, _ = quantize_rtn(float_layer.weight.data, bits=bits, group_size=group_size)

        device = float_layer.weight.device

        qweight = qweight.contiguous()
        zero_point = zero_point.contiguous()

        qweight = qweight.to(device)
        scale = scale.to(device)
        zero_point = zero_point.to(device)

        q_layer = cls(float_layer.in_features, float_layer.out_features, bits, group_size, bias=float_layer.bias is not None, dtype=torch.bfloat16)

        q_layer.qweight.copy_(qweight)
        q_layer.scale.copy_(scale)
        q_layer.zero_point.copy_(zero_point)
        if float_layer.bias is not None:
            q_layer.bias.data.copy_(float_layer.bias.data)

        return q_layer

    def __repr__(self):
        return (f"QuantizedLinear(in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, bits={self.bits}, group_size={self.group_size})")