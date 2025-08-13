import torch

def _quantize_8bit(tensor, scale, maxq):
    """Quantizes a tensor to 8-bit."""
    q_tensor = (tensor / scale).round().clamp(-maxq, maxq)
    q_tensor = (q_tensor + maxq).to(torch.uint8)
    return q_tensor

def _quantize_4bit(tensor, scale, maxq):
    """Quantizes a tensor to 4-bit and packs two values into a uint8."""
    q_tensor = (tensor / scale).round().clamp(-maxq, maxq)
    q_tensor = (q_tensor + maxq).to(torch.uint8)
    
    # Pack two 4-bit values into one uint8 byte
    # The first value is stored in the lower 4 bits, the second in the higher 4 bits.
    packed_tensor = q_tensor[:, :, ::2] | (q_tensor[:, :, 1::2] << 4)
    return packed_tensor

def quantize_rtn(tensor, bits=8, group_size=-1, eps=1e-5):
    """
    Symmetric quantization with optional group-wise support.
    Dispatches to the correct quantization function based on bit width.
    """
    assert bits in [4, 8], "Only 4-bit and 8-bit quantization are supported."
    if bits == 4 and tensor.shape[-1] % 2 != 0:
        raise ValueError("For 4-bit quantization, the last dimension must be even.")

    maxq = 2 ** (bits - 1) - 1

    # --- Reshape for group-wise quantization ---
    shape = tensor.shape
    if group_size != -1:
        if shape[-1] % group_size != 0:
            raise ValueError("The last dimension must be divisible by group_size.")
        t_view = tensor.view(shape[0], -1, group_size)
    else:
        # Treat the entire row as a single group
        t_view = tensor.view(shape[0], 1, shape[-1])

    # --- Compute scale ---
    max_val = t_view.abs().max(dim=2, keepdim=True)[0]
    scale = (max_val / maxq).clamp(min=eps)

    # --- Dispatch to the correct quantization function ---
    if bits == 8:
        q_view = _quantize_8bit(t_view, scale, maxq)
        q_tensor = q_view.view(*shape)
    else: # bits == 4
        q_view = _quantize_4bit(t_view, scale, maxq)
        q_tensor = q_view.view(shape[0], -1)

    zero_point = torch.full_like(scale, maxq)
    
    # Adjust shape for per-row case
    if group_size == -1:
        scale = scale.squeeze(1)
        zero_point = zero_point.squeeze(1)

    return q_tensor, scale, zero_point, group_size
