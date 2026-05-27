import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# Triton kernels are happiest with int32 indices/counters for this kind of routing.
INDEX_DTYPE = torch.int32
COUNT_DTYPE = torch.int32
MASK_DTYPE = torch.int8


if HAS_TRITON:
    @triton.jit
    def capacity_enforce_kernel(
        expert_indices_ptr,
        token_ranks_ptr,
        accepted_counts_ptr,
        overflow_counts_ptr,
        expert_counts_ptr,
        capacity_ptr,
        valid_mask_ptr,
        n_assignments,
        drop_tokens: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_assignments

        # Flat assignment list: one entry per routed token-expert assignment.
        expert = tl.load(expert_indices_ptr + offsets, mask=mask, other=0).to(tl.int32)

        # Unique rank per expert, in arrival order.
        rank = tl.atomic_add(expert_counts_ptr + expert, 1, mask=mask).to(tl.int32)

        cap = tl.load(capacity_ptr + expert, mask=mask, other=0).to(tl.int32)

        if drop_tokens:
            valid = rank < cap
            overflow = mask & (~valid)
            tl.atomic_add(overflow_counts_ptr + expert, 1, mask=overflow)
        else:
            valid = mask

        tl.atomic_add(accepted_counts_ptr + expert, 1, mask=mask & valid)

        tl.store(token_ranks_ptr + offsets, rank, mask=mask)
        tl.store(valid_mask_ptr + offsets, valid.to(tl.int8), mask=mask)


def enforce_capacity_triton(
    expert_indices: torch.Tensor,
    capacity: torch.Tensor,
    drop_tokens: bool,
):
    """
    expert_indices: int tensor of shape [T, K] or flat [num_assignments]
    capacity:        int tensor of shape [num_experts]
    returns:
        valid_mask       – same shape as expert_indices, bool
        accepted_counts  – [num_experts]
        raw_counts       – [num_experts]
        overflow_counts  – [num_experts]
        token_ranks      – same shape as expert_indices
    """
    if not HAS_TRITON or not expert_indices.is_cuda:
        raise RuntimeError("Triton kernels require Triton and CUDA tensors.")

    if expert_indices.dtype not in (torch.int32, torch.int64):
        expert_indices = expert_indices.to(INDEX_DTYPE)
    else:
        expert_indices = expert_indices.to(INDEX_DTYPE)

    if capacity.dtype not in (torch.int32, torch.int64):
        capacity = capacity.to(COUNT_DTYPE)
    else:
        capacity = capacity.to(COUNT_DTYPE)

    # Save original shape so we can restore [T, K] on return.
    original_shape = expert_indices.shape
    expert_indices = expert_indices.reshape(-1).contiguous()
    capacity = capacity.contiguous()

    n_assignments = expert_indices.numel()
    n_experts = capacity.numel()

    # Optional host-side validation; cheap compared with debugging undefined routing.
    if torch.any(expert_indices < 0) or torch.any(expert_indices >= n_experts):
        raise ValueError("expert_indices contains out-of-range expert ids.")

    token_ranks = torch.empty(n_assignments, device=expert_indices.device, dtype=INDEX_DTYPE)
    valid_mask = torch.empty(n_assignments, device=expert_indices.device, dtype=MASK_DTYPE)

    accepted_counts = torch.zeros(n_experts, device=expert_indices.device, dtype=COUNT_DTYPE)
    overflow_counts = torch.zeros(n_experts, device=expert_indices.device, dtype=COUNT_DTYPE)
    raw_counts = torch.zeros(n_experts, device=expert_indices.device, dtype=COUNT_DTYPE)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_assignments, BLOCK_SIZE),)

    capacity_enforce_kernel[grid](
        expert_indices,
        token_ranks,
        accepted_counts,
        overflow_counts,
        raw_counts,
        capacity,
        valid_mask,
        n_assignments,
        drop_tokens=drop_tokens,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return (
        valid_mask.view(original_shape).to(torch.bool),
        accepted_counts,
        raw_counts,
        overflow_counts,
        token_ranks.view(original_shape),
    )


if HAS_TRITON:
    @triton.jit
    def triton_moe_gather_kernel(
        x_ptr,
        flat_buffer_ptr,
        buffer_idx_ptr,
        token_idx_ptr,
        n_assignments,
        flat_dim: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_d = tl.program_id(1)

        m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

        mask_m = m_offs < n_assignments
        mask_d = d_offs < flat_dim

        buf_idx = tl.load(buffer_idx_ptr + m_offs, mask=mask_m, other=0).to(tl.int32)
        tok_idx = tl.load(token_idx_ptr + m_offs, mask=mask_m, other=0).to(tl.int32)

        src_ptrs = x_ptr + tok_idx[:, None] * flat_dim + d_offs[None, :]
        dst_ptrs = flat_buffer_ptr + buf_idx[:, None] * flat_dim + d_offs[None, :]

        mask2d = mask_m[:, None] & mask_d[None, :]

        val = tl.load(src_ptrs, mask=mask2d, other=0.0)
        tl.store(dst_ptrs, val, mask=mask2d)


    @triton.jit
    def triton_moe_scatter_kernel(
        out_padded_ptr,
        active_outputs_ptr,
        buffer_idx_ptr,
        valid_mask_ptr,
        n_assignments,
        flat_dim: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_d = tl.program_id(1)

        m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

        mask_m = m_offs < n_assignments
        mask_d = d_offs < flat_dim

        buf_idx = tl.load(buffer_idx_ptr + m_offs, mask=mask_m, other=0).to(tl.int32)
        valid = tl.load(valid_mask_ptr + m_offs, mask=mask_m, other=0).to(tl.int32)

        src_ptrs = out_padded_ptr + buf_idx[:, None] * flat_dim + d_offs[None, :]
        dst_ptrs = active_outputs_ptr + m_offs[:, None] * flat_dim + d_offs[None, :]

        mask2d = mask_m[:, None] & mask_d[None, :]
        val = tl.load(src_ptrs, mask=mask2d, other=0.0)
        val = val * valid[:, None]
        tl.store(dst_ptrs, val, mask=mask2d)


def moe_gather_triton(
    x: torch.Tensor,
    buffer_idx: torch.Tensor,
    token_idx: torch.Tensor,
    buffer_slots: int,
):
    """
    x:           [num_tokens, hidden_dim]
    buffer_idx:   flat destination slot per assignment
    token_idx:    flat source token per assignment
    buffer_slots: total padded destination rows (explicitly passed)
    """
    if not HAS_TRITON or not x.is_cuda:
        raise RuntimeError("Triton kernels require Triton and CUDA tensors.")

    x = x.contiguous()
    buffer_idx = buffer_idx.reshape(-1).to(INDEX_DTYPE).contiguous()
    token_idx = token_idx.reshape(-1).to(INDEX_DTYPE).contiguous()

    flat_dim = x.shape[-1]
    n_assignments = buffer_idx.numel()

    flat_buffer = torch.zeros(
        buffer_slots,
        flat_dim,
        device=x.device,
        dtype=x.dtype,
    )

    BLOCK_M = 128
    BLOCK_D = triton.next_power_of_2(flat_dim) if flat_dim < 128 else 128
    grid = (triton.cdiv(n_assignments, BLOCK_M), triton.cdiv(flat_dim, BLOCK_D))

    triton_moe_gather_kernel[grid](
        x,
        flat_buffer,
        buffer_idx,
        token_idx,
        n_assignments,
        flat_dim=flat_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
    )
    return flat_buffer


def moe_scatter_triton(
    out_padded: torch.Tensor,
    buffer_idx: torch.Tensor,
    valid_mask: torch.Tensor,
):
    """
    out_padded:   [buffer_slots, hidden_dim]
    buffer_idx:    flat source slot per assignment
    valid_mask:    flat boolean/0-1 mask per assignment
    """
    if not HAS_TRITON or not out_padded.is_cuda:
        raise RuntimeError("Triton kernels require Triton and CUDA tensors.")

    out_padded = out_padded.contiguous()
    buffer_idx = buffer_idx.reshape(-1).to(INDEX_DTYPE).contiguous()
    valid_mask = valid_mask.reshape(-1).to(MASK_DTYPE).contiguous()

    flat_dim = out_padded.shape[-1]
    n_assignments = buffer_idx.numel()

    active_outputs = torch.empty(
        n_assignments,
        flat_dim,
        device=out_padded.device,
        dtype=out_padded.dtype,
    )

    BLOCK_M = 128
    BLOCK_D = triton.next_power_of_2(flat_dim) if flat_dim < 128 else 128
    grid = (triton.cdiv(n_assignments, BLOCK_M), triton.cdiv(flat_dim, BLOCK_D))

    triton_moe_scatter_kernel[grid](
        out_padded,
        active_outputs,
        buffer_idx,
        valid_mask,
        n_assignments,
        flat_dim=flat_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
    )
    return active_outputs