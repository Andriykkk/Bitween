import torch

def quantize_rtn(weight, bits=4, group_size=-1):
    """
    Performs Round-To-Nearest (RTN) quantization on a weight tensor.

    Args:
        weight (torch.Tensor): The input weight tensor.
        bits (int): The number of bits for quantization (e.g., 4).
        group_size (int): The size of groups for quantization. 
                          -1 means per-channel quantization.
                          >0 means group-wise quantization.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the
        quantized weights (q_weight), scales, and zero points.
    """
    assert bits <= 8, "Only quantization to 8 bits or less is supported"
    
    # --- Determine quantization range ---
    q_min = 0
    q_max = 2**bits - 1

    # --- Reshape for group-wise or per-channel quantization ---
    orig_shape = weight.shape
    if group_size > 0:
        weight = weight.reshape(-1, group_size)
    else: # Per-channel
        weight = weight.reshape(weight.shape[0], -1)

    # --- Calculate scale and zero point ---
    # The scale is the range of the float values divided by the range of the integer values
    w_max = torch.max(weight, dim=-1, keepdim=True)[0]
    w_min = torch.min(weight, dim=-1, keepdim=True)[0]
    
    # To prevent division by zero
    scale = (w_max - w_min).clamp(min=1e-6) / (q_max - q_min)
    
    # The zero point is the float value that maps to the integer 0
    zero_point = torch.round(w_min / scale)

    # --- Quantize the weights ---
    # 1. Rescale the weights to the integer range
    # 2. Add the zero point
    # 3. Round to the nearest integer
    # 4. Clamp to the quantization range
    q_weight = torch.round(weight / scale) + zero_point
    q_weight = torch.clamp(q_weight, q_min, q_max).to(torch.uint8)

    # --- Reshape back to original ---
    if group_size > 0:
        q_weight = q_weight.reshape(orig_shape)
        scale = scale.reshape(orig_shape[0], -1)
        zero_point = zero_point.reshape(orig_shape[0], -1)
    
    return q_weight, scale, zero_point
