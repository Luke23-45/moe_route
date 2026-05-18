from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from moe_route.models.moe import ExpertMLP, MoEFeedForward
from moe_route.routing.routers import RouterConfig
from moe_route.routing.types import RoutingDiagnostics


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    max_seq_len: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    dropout: float
    moe_enabled: bool
    moe_every_n_layers: int
    num_experts: int
    expert_hidden_size: int
    router: RouterConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, max_seq_len: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout
        
        # Precompute RoPE frequencies
        freqs = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2)[: (self.head_dim // 2)].float() / self.head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, width = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply RoPE
        q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
        k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
        freqs_cis = self.freqs_cis[:seq_len].view(1, 1, seq_len, -1)
        q = torch.view_as_real(q_ * freqs_cis).flatten(3).type_as(q)
        k = torch.view_as_real(k_ * freqs_cis).flatten(3).type_as(k)
        
        # Flash Attention
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True
        )
        
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, width)
        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        use_moe = cfg.moe_enabled and cfg.moe_every_n_layers > 0 and (layer_idx + 1) % cfg.moe_every_n_layers == 0
        if use_moe:
            self.ff = MoEFeedForward(
                d_model=cfg.d_model,
                num_experts=cfg.num_experts,
                expert_hidden_size=cfg.expert_hidden_size,
                dropout=cfg.dropout,
                router_cfg=cfg.router,
            )
        else:
            self.ff = ExpertMLP(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.ln1(x))
        ff = self.ff(self.ln2(x))
        if isinstance(ff, tuple):
            y, aux = ff
        else:
            y = ff
            aux = torch.zeros((), device=x.device)
        return x + y, aux


class DecoderOnlyLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg, i) for i in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        batch, seq_len = input_ids.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.cfg.max_seq_len}.")
        x = self.token_emb(input_ids)
        x = self.drop(x)
        aux_loss = torch.zeros((), device=input_ids.device)
        for block in self.blocks:
            x, aux = block(x)
            aux_loss = aux_loss + aux
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if labels is not None:
            lm_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            loss = lm_loss + aux_loss
        return logits, loss, {"aux_loss": aux_loss.detach()}

    def routing_diagnostics(self) -> list[RoutingDiagnostics]:
        diagnostics: list[RoutingDiagnostics] = []
        for module in self.modules():
            if isinstance(module, MoEFeedForward) and module.last_diagnostics is not None:
                diagnostics.append(module.last_diagnostics)
        return diagnostics

    def pressure_state_dict(self) -> list[list[dict[str, torch.Tensor] | None]]:
        return [m.pressure_state_dict() for m in self.modules() if isinstance(m, MoEFeedForward)]

    def load_pressure_state_dict(self, states: list[list[dict[str, torch.Tensor] | None]]) -> None:
        moe_layers = [m for m in self.modules() if isinstance(m, MoEFeedForward)]
        if len(states) != len(moe_layers):
            raise ValueError(
                f"Pressure state count mismatch: checkpoint has {len(states)}, model has {len(moe_layers)}."
            )
        for module, state in zip(moe_layers, states, strict=True):
            module.load_pressure_state_dict(state)


def build_model_cfg(cfg) -> ModelConfig:
    router_cfg = RouterConfig(
        kind=cfg.router.kind,
        d_model=int(cfg.model.d_model),
        num_experts=int(cfg.model.moe.num_experts),
        top_k=int(cfg.router.top_k),
        capacity_factor=float(cfg.router.capacity_factor),
        drop_tokens=bool(cfg.router.drop_tokens),
        aux_loss_weight=float(cfg.router.aux_loss_weight),
        pressure_lr=float(cfg.router.get("pressure_lr", 0.05)),
        pressure_alpha=float(cfg.router.get("pressure_alpha", 1.0)),
        pressure_beta=float(cfg.router.get("pressure_beta", 1.0)),
        pressure_gamma=float(cfg.router.get("pressure_gamma", 0.0)),
        pressure_decay=float(cfg.router.get("pressure_decay", 0.0)),
    )
    return ModelConfig(
        vocab_size=int(cfg.model.vocab_size),
        max_seq_len=int(cfg.model.max_seq_len),
        d_model=int(cfg.model.d_model),
        n_layers=int(cfg.model.n_layers),
        n_heads=int(cfg.model.n_heads),
        d_ff=int(cfg.model.d_ff),
        dropout=float(cfg.model.dropout),
        moe_enabled=bool(cfg.model.moe.enabled),
        moe_every_n_layers=int(cfg.model.moe.every_n_layers),
        num_experts=int(cfg.model.moe.num_experts),
        expert_hidden_size=int(cfg.model.moe.expert_hidden_size),
        router=router_cfg,
    )
