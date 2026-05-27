"""Triton-based fused kernels for MoE routing.

These kernels replace the O(N log N) PyTorch argsort overhead with an O(N) 
atomic binning approach, matching the theoretical optimization limits of modern
MoE implementations (like DeepSeek-V3 or Megablocks).
"""

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    @triton.jit
    def triton_capacity_enforce_kernel(
        expert_indices_ptr,
        token_ranks_ptr,
        accepted_counts_ptr,
        overflow_counts_ptr,
        raw_load_ptr,
        capacity_ptr,
        valid_mask_ptr,
        num_tokens,
        num_experts,
        drop_tokens: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fuses token binning, capacity checking, and rank assignment into a single pass.
        Replaces torch.argsort, torch.bincount, and torch.cummax.
        """
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_tokens

        # Load the expert index for each token in this block
        # other=0 is safe because it's masked out
        expert_idx = tl.load(expert_indices_ptr + offsets, mask=mask, other=0)
        
        # Atomically increment the count for the assigned expert to get a unique rank
        # This gives each token its rank for its assigned expert.
        token_rank = tl.atomic_add(raw_load_ptr + expert_idx, 1, mask=mask)
        
        # Store the rank
        tl.store(token_ranks_ptr + offsets, token_rank, mask=mask)
        
        # Load capacities for the assigned experts
        cap = tl.load(capacity_ptr + expert_idx, mask=mask, other=0)
        
        if drop_tokens:
            is_valid = token_rank < cap
            is_overflow = token_rank >= cap
            
            tl.atomic_add(overflow_counts_ptr + expert_idx, 1, mask=mask & is_overflow)
        else:
            is_valid = token_rank >= 0  # Always true
            
        tl.atomic_add(accepted_counts_ptr + expert_idx, 1, mask=mask & is_valid)
        
        # Store valid mask (cast to int8 as it corresponds to torch.bool)
        tl.store(valid_mask_ptr + offsets, is_valid.to(tl.int8), mask=mask)

def enforce_capacity_triton(expert_indices: torch.Tensor, capacity: torch.Tensor, drop_tokens: bool):
    """
    Python wrapper for the Triton capacity enforcement kernel.
    Returns: (valid_mask, accepted_load, raw_load, overflow, token_ranks)
    """
    if not HAS_TRITON or not expert_indices.is_cuda:
        raise RuntimeError("Triton kernels require Triton and CUDA tensors.")
    
    num_tokens = expert_indices.numel()
    num_experts = capacity.numel()
    
    expert_indices_flat = expert_indices.reshape(-1).contiguous()
    
    token_ranks = torch.empty_like(expert_indices_flat, dtype=torch.long)
    valid_mask = torch.empty_like(expert_indices_flat, dtype=torch.bool)
    
    accepted_counts = torch.zeros(num_experts, dtype=torch.long, device=expert_indices.device)
    overflow_counts = torch.zeros(num_experts, dtype=torch.long, device=expert_indices.device)
    raw_load = torch.zeros(num_experts, dtype=torch.long, device=expert_indices.device)
    
    # Grid configuration
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(num_tokens, BLOCK_SIZE),)
    
    triton_capacity_enforce_kernel[grid](
        expert_indices_flat,
        token_ranks,
        accepted_counts,
        overflow_counts,
        raw_load,
        capacity,
        valid_mask,
        num_tokens,
        num_experts,
        drop_tokens=drop_tokens,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (
        valid_mask.reshape_as(expert_indices), 
        accepted_counts, 
        raw_load, 
        overflow_counts, 
        token_ranks.reshape_as(expert_indices)
    )
