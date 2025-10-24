import torch
import torch.nn as nn
import torch.nn.functional as F
from bitween.functional import quantize_rtn
from .kernels.int import quantized_linear_kernel
import triton
import triton.language as tl
import copy

@triton.jit
def dequantize_weights_kernel(
    qweight_ptr,    # [N, packed_K] int32
    scale_ptr,      # [N, num_groups] fp16
    zp_ptr,         # [N, num_groups] fp16
    weight_ptr,     # [N, K] fp16 output
    N, K,
    packed_K,
    num_groups,
    bits: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel to dequantize weights from INT to FP16."""
    pid = tl.program_id(0)
    n = pid

    if n >= N:
        return

    values_per_int32: tl.constexpr = 32 // bits
    mask: tl.constexpr = (1 << bits) - 1

    # Process K dimension in chunks
    for k_start in range(0, K, BLOCK_SIZE):
        k_offs = k_start + tl.arange(0, BLOCK_SIZE)
        k_mask = k_offs < K

        # Get packed indices
        packed_k = k_offs // values_per_int32
        idx_in_packed = k_offs % values_per_int32

        # Load packed values
        packed_vals = tl.load(qweight_ptr + n * packed_K + packed_k, mask=k_mask, other=0)

        # Extract quantized values
        shift = idx_in_packed * bits
        q_vals = (packed_vals >> shift) & mask

        # Get scale and zero_point for this group
        group_idx = k_offs // group_size
        scales = tl.load(scale_ptr + n * num_groups + group_idx, mask=k_mask, other=0.0)
        zps = tl.load(zp_ptr + n * num_groups + group_idx, mask=k_mask, other=0.0)

        # Dequantize: weight = scale * (q - zero_point)
        weights = scales * (q_vals.to(tl.float32) - zps)

        # Store
        tl.store(weight_ptr + n * K + k_offs, weights.to(tl.float16), mask=k_mask)

class QuantizedLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, qweight, scale, zero_point, bias, bits, group_size):
        # Save tensors for backward pass
        ctx.save_for_backward(x, qweight, scale, zero_point, bias)

        ctx.bits = bits
        ctx.group_size = group_size

        device = x.device
        original_shape = x.shape
        target_dtype = torch.float16

        # Convert input to target dtype before kernel
        x_reshaped = x.reshape(-1, original_shape[-1]).to(target_dtype)

        M, K = x_reshaped.shape
        N, packed_K = qweight.shape

        # BASELINE: Dequantize to FP16, then use PyTorch's cuBLAS matmul
        values_per_int32 = 32 // bits
        actual_K = packed_K * values_per_int32
        num_groups = actual_K // group_size

        # Allocate temporary dequantized weights
        weight_fp16 = torch.empty((N, actual_K), dtype=torch.float16, device=device)

        # Launch Triton dequantization kernel
        grid = lambda meta: (N,)
        dequantize_weights_kernel[grid](
            qweight, scale, zero_point, weight_fp16,
            N, actual_K, packed_K, num_groups,
            bits=bits,
            group_size=group_size,
            BLOCK_SIZE=256
        )

        c = torch.matmul(x_reshaped, weight_fp16.T)

        # Add bias
        if bias is not None:
            c = c + bias

        # Clear dequantized weights immediately
        del weight_fp16

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
            grad_input = grad_output @ weight.to(grad_output.dtype)

        return grad_input, None, None, None, None, None, None

class QuantizedLinear(nn.Module):
    """
    A quantized linear layer module that supports both 4-bit and 8-bit weights.
    """
    def __init__(self, in_features, out_features, bits=8, group_size=32, bias=True, dtype=torch.float16):
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

        # if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 and dtype == torch.bfloat16:
        #     self.dtype = torch.bfloat16
        # else:
        #     self.dtype = torch.float16
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

        q_layer = cls(float_layer.in_features, float_layer.out_features, bits, group_size, bias=float_layer.bias is not None, dtype=torch.float16)

        q_layer = q_layer.to(device)

        q_layer.qweight.copy_(qweight)
        q_layer.scale.copy_(scale)
        q_layer.zero_point.copy_(zero_point)
        if float_layer.bias is not None:
            q_layer.bias.data.copy_(float_layer.bias.data)

        return q_layer

    def __repr__(self):
        return (f"QuantizedLinear(in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, bits={self.bits}, group_size={self.group_size})")