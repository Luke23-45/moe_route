from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PressureStateConfig:
    num_experts: int
    lr: float = 0.05
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 0.0
    decay: float = 0.0
    capacity_weights: tuple[float, ...] | None = None


class PressureState:
    """Reflected nonnegative expert-pressure state."""

    def __init__(self, cfg: PressureStateConfig, device: torch.device | str = "cpu") -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        if cfg.capacity_weights is None:
            capacity = torch.full((cfg.num_experts,), 1.0 / cfg.num_experts, device=self.device)
        else:
            capacity = torch.tensor(cfg.capacity_weights, dtype=torch.float32, device=self.device)
            capacity = capacity / capacity.sum().clamp_min(1e-8)
        self.capacity_weights = capacity
        self.q = torch.zeros(cfg.num_experts, dtype=torch.float32, device=self.device)

    def to(self, device: torch.device | str) -> PressureState:
        self.device = torch.device(device)
        self.q = self.q.to(self.device)
        self.capacity_weights = self.capacity_weights.to(self.device)
        return self

    def penalty(self) -> torch.Tensor:
        denom = self.capacity_weights.clamp_min(1e-8).pow(self.cfg.beta)
        prior = self.cfg.gamma * self.capacity_weights.clamp_min(1e-8).log()
        return self.cfg.alpha * self.q / denom - prior

    @torch.no_grad()
    def update(self, load_fraction: torch.Tensor) -> None:
        load = load_fraction.detach().to(self.q.device, dtype=self.q.dtype)
        delta = load - self.capacity_weights
        if self.cfg.decay:
            self.q.mul_(1.0 - self.cfg.decay)
        self.q.add_(self.cfg.lr * delta)
        self.project_()

    @torch.no_grad()
    def project_(self) -> None:
        self.q.clamp_(min=0.0)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"q": self.q.detach().clone(), "capacity_weights": self.capacity_weights.detach().clone()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.q = state["q"].detach().clone().to(self.device)
        self.capacity_weights = state["capacity_weights"].detach().clone().to(self.device)
        self.project_()

    def reset(self) -> None:
        self.q.zero_()
