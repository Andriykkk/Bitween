import torch

@torch.no_grad()
def quantize_rtn(weight: torch.Tensor, bits: int = 8, group_size: int = 32):
    weight = weight.contiguous()
    assert bits in [2, 4, 8], "Only 2, 4, or 8-bit quantization supported."
    
    Qmin = 0
    Qmax = (1 << bits) - 1

    out_features, in_features = weight.shape
    pad = (group_size - in_features % group_size) % group_size
    # if pad > 0:
    #     weight = torch.cat([weight, torch.zeros(out_features, pad, device=weight.device, dtype=weight.dtype)], dim=1)
    #     in_features += pad
    if pad > 0:
        raise ValueError(f"Padding {pad} columns to {in_features} to make it a multiple of {group_size}")

    num_groups = in_features // group_size
    weight = weight.view(out_features, num_groups, group_size)

    w_min = weight.amin(dim=-1, keepdim=True)
    w_max = weight.amax(dim=-1, keepdim=True)

    scale = ((w_max - w_min) / Qmax).clamp(min=1e-8)
    zero_point = torch.round(-w_min / scale).clamp(Qmin, Qmax).to(torch.int32)

    # noise = torch.empty_like(q).uniform_(-0.5, 0.5) noise = torch.zeros_like(q)
    q = torch.round(weight / scale + zero_point).clamp(Qmin, Qmax).to(torch.int32)

    values_per_int32 = 32 // bits
    total_vals = out_features * num_groups * group_size
    total_int32 = total_vals // values_per_int32

    q_flat = q.reshape(-1)
    q_packed = torch.zeros(total_int32, dtype=torch.int32, device=weight.device)

    for i in range(values_per_int32):
        q_part = q_flat[i::values_per_int32]
        q_packed |= (q_part << (i * bits))

    packed_per_row = (num_groups * group_size) // values_per_int32
    q_packed = q_packed.view(out_features, packed_per_row)

    return q_packed, scale.squeeze(-1), zero_point.squeeze(-1), group_size

@torch.no_grad()
def dequantize_rtn(q_packed: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, group_size: int, bits: int = 8):
    assert bits in [2, 4, 8], "Only 2, 4, or 8-bit quantization supported."

    values_per_int32 = 32 // bits
    out_features, packed_per_row = q_packed.shape
    num_groups = (packed_per_row * values_per_int32) // group_size
    in_features = num_groups * group_size

    q_flat = torch.empty(out_features * num_groups * group_size, dtype=torch.int32, device=q_packed.device)

    for i in range(values_per_int32):
        shift = i * bits
        mask = (1 << bits) - 1
        q_vals = (q_packed.view(-1) >> shift) & mask
        q_flat[i::values_per_int32] = q_vals

    q = q_flat.view(out_features, num_groups, group_size)

    scale = scale.view(out_features, num_groups, 1)
    zero_point = zero_point.view(out_features, num_groups, 1)

    # Dequantization formula: w = scale * (q - zp)
    weight = scale * (q.float() - zero_point.float())
    return weight.view(out_features, in_features)