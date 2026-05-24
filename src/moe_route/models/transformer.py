from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from moe_route.models.moe import ExpertMLP, MoEFeedForward
from moe_route.routing.routers import RouterConfig
from moe_route.routing.types import RoutingDiagnostics


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


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
        assert self.head_dim % 2 == 0, (
            f"RoPE requires even head_dim, got {self.head_dim} "
            f"(d_model={d_model}, n_heads={n_heads})"
        )
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout
        
        # Precompute RoPE frequencies
        freqs = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2)[: (self.head_dim // 2)].float() / self.head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        freqs = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, width = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply RoPE (Inductor friendly, no complex numbers)
        cos = self.cos_cached[:seq_len].view(1, 1, seq_len, self.head_dim).to(q.dtype)
        sin = self.sin_cached[:seq_len].view(1, 1, seq_len, self.head_dim).to(q.dtype)
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        
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
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the model with GPT-style small weights.

        PyTorch defaults are too large for a tied embedding/lm_head setup:
        `nn.Embedding` starts with unit-std weights and `nn.Linear` defaults to
        Kaiming/uniform init. In a decoder LM this can produce extreme random
        logits at step 0, so we switch to the standard small-normal
        initialization used by GPT-family models and scale residual output
        projections by depth.
        """
        init_std = 0.02
        residual_std = init_std / math.sqrt(2 * self.cfg.n_layers)

        def _init_module(module: nn.Module) -> None:
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
            elif isinstance(module, nn.Linear):
                if module is self.lm_head and module.weight is self.token_emb.weight:
                    # The tied LM head shares token_emb.weight, which is already initialized.
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                    return
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        self.apply(_init_module)

        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=residual_std)
            if block.attn.proj.bias is not None:
                nn.init.zeros_(block.attn.proj.bias)

            if isinstance(block.ff, ExpertMLP):
                out_proj = block.ff.net[-1]
                nn.init.normal_(out_proj.weight, mean=0.0, std=residual_std)
                if out_proj.bias is not None:
                    nn.init.zeros_(out_proj.bias)
            elif isinstance(block.ff, MoEFeedForward):
                nn.init.normal_(block.ff.experts.w2, mean=0.0, std=residual_std)
                nn.init.zeros_(block.ff.experts.b2)
                if block.ff.shared_experts is not None:
                    nn.init.normal_(block.ff.shared_experts.w2, mean=0.0, std=residual_std)
                    nn.init.zeros_(block.ff.shared_experts.b2)

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
            if labels.dtype != torch.long:
                labels = labels.long()
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
        top_k=int(cfg.router.get("top_k", 1)),
        capacity_factor=float(cfg.router.get("capacity_factor", 1.25)),
        drop_tokens=bool(cfg.router.get("drop_tokens", True)),
        aux_loss_weight=float(cfg.router.get("aux_loss_weight", 0.01)),
        z_loss_weight=float(cfg.router.get("z_loss_weight", 0.001)),
        pressure_lr=float(cfg.router.get("pressure_lr", 0.05)),
        pressure_beta=float(cfg.router.get("pressure_beta", 1.0)),
        pressure_decay=float(cfg.router.get("pressure_decay", 0.0)),
        pressure_warmup_steps=int(cfg.router.get("pressure_warmup_steps", 1000)),
        # Reflected controller (v2) specific fields
        temperature=float(cfg.router.get("temperature", 1.0)),
        pressure_scale=float(cfg.router.get("pressure_scale", 1.0)),
        pressure_eps=float(cfg.router.get("pressure_eps", 1e-8)),
        learnable_bias=bool(cfg.router.get("learnable_bias", True)),
        shared_experts=int(cfg.router.get("shared_experts", 0)),
        routing_mode=str(cfg.router.get("routing_mode", "dense")),
        gate_function=str(cfg.router.get("gate_function", "softmax")),
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
