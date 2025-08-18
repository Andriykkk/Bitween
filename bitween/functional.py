import torch

@torch.no_grad()
def quantize_rtn(weight: torch.Tensor, bits: int = 8, group_size: int = 32, print_paddings: bool = False):
    weight = weight.contiguous()

    assert bits in [2, 4, 8], "Only 2, 4, or 8-bit quantization supported."
    
    Qmin = 0
    Qmax = (1 << bits) - 1

    out_features, in_features = weight.shape

    # Pad in_features to be multiple of group_size if needed
    pad = (group_size - in_features % group_size) % group_size
    # if print_paddings:
    #     print(f"Padding {pad} columns to {in_features} to make it a multiple of {group_size}")
    # if pad > 0:
    #     weight = torch.cat([weight, torch.zeros(out_features, pad, device=weight.device, dtype=weight.dtype)], dim=1)
    #     in_features += pad
    if pad > 0:
        raise ValueError(f"Padding {pad} columns to {in_features} to make it a multiple of {group_size}")

    num_groups = in_features // group_size
    weight = weight.view(out_features, num_groups, group_size)

    w_min = weight.amin(dim=-1, keepdim=True)  # [out_features, num_groups, 1]
    w_max = weight.amax(dim=-1, keepdim=True)

    scale = ((w_max - w_min) / Qmax).clamp(min=1e-8)  # [out, groups, 1]
    # Compute zero_point (per group)
    zero_point = torch.round(Qmin - w_min / scale).clamp(Qmin, Qmax).to(torch.int32)

    # Quantize with RTN
    q = ((weight - zero_point) / scale)
    noise = torch.empty_like(q).uniform_(-0.5, 0.5)
    q = (q + noise).round().clamp(Qmin, Qmax).to(torch.int32)  # still [out, groups, group_size]

    # Pack into int32
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

    print(q_packed.shape, scale.squeeze(-1).shape, zero_point.squeeze(-1).shape)
    return q_packed, scale.squeeze(-1), zero_point.squeeze(-1), group_size
