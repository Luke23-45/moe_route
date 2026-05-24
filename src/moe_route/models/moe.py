from __future__ import annotations

import torch
from torch import nn
from moe_route.utils.compile_state import conditional_dynamo_disable

from moe_route.routing.routers import Router, RouterConfig, build_router
from moe_route.routing.types import RoutingDiagnostics
from moe_route.routing.reflected_controller import ReflectedController


class ExpertMLP(nn.Module):
    def __init__(self, d_model: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BatchedExpertMLP(nn.Module):
    def __init__(self, num_experts: int, d_model: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(num_experts, d_model, hidden_size))
        self.b1 = nn.Parameter(torch.empty(num_experts, 1, hidden_size))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_size, d_model))
        self.b2 = nn.Parameter(torch.empty(num_experts, 1, d_model))
        self.dropout = nn.Dropout(dropout)

        import math
        for i in range(num_experts):
            # w1[i] is [D, H] used as x @ w1 (transposed relative to nn.Linear convention).
            # PyTorch's _calculate_fan_in_and_fan_out assumes [out, in] layout, so it
            # would pick fan_in=H. The actual fan_in is D (shape[0]).
            # Using mode='fan_out' selects shape[0]=D as the fan dimension.
            nn.init.kaiming_uniform_(self.w1[i], a=math.sqrt(5), mode='fan_out')
            fan_in_1 = self.w1[i].shape[0]  # D = actual input dimension
            bound_1 = 1 / math.sqrt(fan_in_1) if fan_in_1 > 0 else 0
            nn.init.uniform_(self.b1[i], -bound_1, bound_1)

            # w2[i] is [H, D] used as h @ w2. Actual fan_in is H (shape[0]).
            nn.init.kaiming_uniform_(self.w2[i], a=math.sqrt(5), mode='fan_out')
            fan_in_2 = self.w2[i].shape[0]  # H = actual input dimension
            bound_2 = 1 / math.sqrt(fan_in_2) if fan_in_2 > 0 else 0
            nn.init.uniform_(self.b2[i], -bound_2, bound_2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.bmm(x, self.w1) + self.b1
        h = torch.nn.functional.gelu(h)
        h = self.dropout(h)
        return torch.bmm(h, self.w2) + self.b2

    def forward_shared_sum(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all experts to all tokens and sum expert outputs."""
        num_experts = self.w1.shape[0]
        expanded = x.unsqueeze(0).expand(num_experts, -1, -1)
        return self.forward(expanded).sum(dim=0)

    def forward_dense_fused(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """SOTA Fused Grouped GEMM for Dense Soft Routing.
        
        Mathematically computes: y_t = Σ_e p_{t,e} f_e(x_t)
        But avoids expanding x to [E, T, D] and fuses the reduction into the W2 matmul.
        
        Args:
            x: [T, d_model] flattened tokens.
            weights: [T, E] routing probabilities.
            
        Returns:
            [T, d_model] output.
        """
        # Step 1: W1 Projection without expansion
        # x: [T, D], w1: [E, D, H] -> h: [E, T, H]
        T, D = x.shape
        E, _, H = self.w1.shape
        w1_reshaped = self.w1.transpose(0, 1).reshape(D, E * H)
        h = torch.matmul(x, w1_reshaped).view(T, E, H).transpose(0, 1) + self.b1
        h = torch.nn.functional.gelu(h)
        h = self.dropout(h)

        # Step 2: Weight hidden states by routing probabilities BEFORE W2
        # weights: [T, E] -> [E, T, 1]
        p = weights.T.unsqueeze(-1)
        h_weighted = h * p  # [E, T, H]

        # Step 3: Fused W2 Projection & Expert Reduction
        # h_weighted: [E, T, H], w2: [E, H, D] -> out: [T, D]
        # Using bmm is faster than einsum
        out = torch.bmm(h_weighted, self.w2).sum(dim=0)

        # Step 4: Add weighted biases
        # p_raw: [T, E], b2: [E, 1, D] -> [T, D]
        bias_term = torch.matmul(weights, self.b2.squeeze(1))
        
        return out + bias_term


class MoEFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        expert_hidden_size: int,
        dropout: float,
        router_cfg: RouterConfig,
    ) -> None:
        super().__init__()
        self.router: Router = build_router(router_cfg)
        self.experts = BatchedExpertMLP(num_experts, d_model, expert_hidden_size, dropout)
        self.shared_expert_count = 0
        self.shared_experts: BatchedExpertMLP | None = None
        if isinstance(self.router, ReflectedController):
            self.shared_expert_count = max(int(getattr(self.router.cfg, "shared_experts", 0)), 0)
            if self.shared_expert_count > 0:
                self.shared_experts = BatchedExpertMLP(
                    self.shared_expert_count,
                    d_model,
                    expert_hidden_size,
                    dropout,
                )
        self.last_diagnostics: RoutingDiagnostics | None = None
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Dispatch to dense or sparse path based on router type and explicit mode
        if isinstance(self.router, ReflectedController) and getattr(self.router.cfg, "routing_mode", "dense") == "dense":
            return self._forward_dense(x)
        return self._forward_sparse(x)

    def _forward_dense(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Dense reflected routing: every token → every expert, weighted by soft probs.

        Implements §1 line 75-78: y_t = Σ_e p̃_{t,e} · f_e(h_t)
        """
        original_shape = x.shape  # [B, S, D]
        flat = x.reshape(-1, original_shape[-1])  # [T, D]
        route = self.router(flat)

        # Use SOTA fused Grouped GEMM to avoid [E, T, D] memory allocation
        output = self.experts.forward_dense_fused(flat, route.combine_weights)  # [T, D]
        output = output + self._shared_output(flat)

        self.last_diagnostics = route.diagnostics
        return output.reshape(original_shape), route.diagnostics.aux_loss

    @conditional_dynamo_disable
    def _forward_sparse(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sparse top-k routing with scatter/gather dispatch (original path)."""
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        route = self.router(flat)

        num_tokens, top_k = route.expert_indices.shape
        flat_indices = route.expert_indices.view(-1)
        flat_mask = route.dispatch_mask.view(-1)
        token_ranks = route.token_ranks.view(-1)

        cap = self.router.capacity.capacity_per_expert(num_tokens)
        flat_dim = flat.shape[-1]

        active_positions = flat_mask.nonzero(as_tuple=False).squeeze(1)
        active_outputs = torch.zeros(num_tokens * top_k, flat_dim, dtype=flat.dtype, device=flat.device)

        if active_positions.numel() > 0:
            active_experts = flat_indices.index_select(0, active_positions)
            active_ranks = token_ranks.index_select(0, active_positions)
            active_tokens = torch.div(active_positions, top_k, rounding_mode="floor")
            active_buffer_idx = active_experts * cap + active_ranks

            flat_buffer = torch.zeros(
                self.num_experts * cap,
                flat_dim,
                dtype=flat.dtype,
                device=flat.device,
            )
            flat_buffer.index_copy_(0, active_buffer_idx, flat.index_select(0, active_tokens))

            buffer = flat_buffer.view(self.num_experts, cap, flat_dim)
            buffer_out = self.experts(buffer)
            expert_out = buffer_out.view(self.num_experts * cap, flat_dim).index_select(0, active_buffer_idx)
            active_outputs.index_copy_(0, active_positions, expert_out)

        weights = route.combine_weights.view(-1)
        output = (active_outputs * weights.unsqueeze(-1)).view(num_tokens, top_k, flat_dim)
        if top_k == 1:
            output = output.squeeze(1)
        else:
            output = output.sum(dim=1)
        output = output + self._shared_output(flat)

        self.last_diagnostics = route.diagnostics
        return output.reshape(original_shape), route.diagnostics.aux_loss

    def _shared_output(self, flat: torch.Tensor) -> torch.Tensor:
        if self.shared_experts is None:
            return torch.zeros_like(flat)
        return self.shared_experts.forward_shared_sum(flat)

    def pressure_state_dict(self) -> list[dict[str, torch.Tensor] | None]:
        return [self.router.pressure_state_dict()]

    def load_pressure_state_dict(self, states: list[dict[str, torch.Tensor] | None]) -> None:
        if states:
            self.router.load_pressure_state_dict(states[0])

