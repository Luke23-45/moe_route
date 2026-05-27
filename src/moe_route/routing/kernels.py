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
        expert_idx = tl.load(expert_indices_ptr + offsets, mask=mask, other=-1)
        
        # We need to atomically increment the count for the assigned expert to get a unique rank
        # Since Triton doesn't natively support atomic_add returning the *old* value in all backends seamlessly,
        # we do a loop or just rely on atomic_add behavior if supported.
        # Actually, tl.atomic_add returns the old value!
        token_rank = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
        
        for i in range(BLOCK_SIZE):
            if offsets[i] < num_tokens:
                e_idx = tl.load(expert_indices_ptr + offsets[i])
                # Atomically add 1 to the raw load for this expert
                old_rank = tl.atomic_add(raw_load_ptr + e_idx, 1)
                
                # We can't easily write to a tensor from inside this scalar loop directly,
                # but we can write directly to global memory for the token_rank.
                tl.store(token_ranks_ptr + offsets[i], old_rank)
                
                cap = tl.load(capacity_ptr + e_idx)
                is_valid = True
                if drop_tokens:
                    if old_rank >= cap:
                        is_valid = False
                        tl.atomic_add(overflow_counts_ptr + e_idx, 1)
                
                if is_valid:
                    tl.atomic_add(accepted_counts_ptr + e_idx, 1)
                    tl.store(valid_mask_ptr + offsets[i], 1)
                else:
                    tl.store(valid_mask_ptr + offsets[i], 0)

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
