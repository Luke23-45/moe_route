from __future__ import annotations

import torch


def coefficient_of_variation(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x.float().std(unbiased=False) / x.float().mean().clamp_min(eps)


def gini(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    values = x.float().flatten()
    if values.numel() == 0:
        return torch.zeros((), device=x.device)
    sorted_values = values.sort().values
    n = sorted_values.numel()
    index = torch.arange(1, n + 1, device=x.device, dtype=sorted_values.dtype)
    total = sorted_values.sum().clamp_min(eps)
    return (2 * (index * sorted_values).sum() / (n * total)) - ((n + 1) / n)


def routing_entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(dim=-1).mean()

