from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from moe_route.models.transformer import DecoderOnlyLM, build_model_cfg
from moe_route.routing.types import RoutingDiagnostics


ROOT = Path(__file__).resolve().parents[1]


def load_composed_cfg(router_cfg_path: str, model_cfg_path: str):
    cfg = OmegaConf.load(ROOT / "configs" / "config.yaml")
    cfg.router = OmegaConf.load(ROOT / router_cfg_path)
    cfg.model = OmegaConf.load(ROOT / model_cfg_path)
    cfg.data = OmegaConf.load(ROOT / "configs" / "data" / "tinystories.yaml")
    cfg.tokenizer = OmegaConf.load(ROOT / "configs" / "tokenizer" / "byte.yaml")
    return cfg


def gpt_style_small_init(model: torch.nn.Module, std: float = 0.02) -> None:
    """Reinitialize the shared transformer stack to a small-logit regime.

    This is only for diagnosis. It demonstrates whether the startup loss spike
    comes from the shared parameter scale rather than from a specific router.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.LayerNorm):
            if module.weight is not None:
                torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    for name, parameter in model.named_parameters():
        if "experts.w1" in name or "experts.w2" in name:
            torch.nn.init.normal_(parameter, mean=0.0, std=std)
        elif "experts.b1" in name or "experts.b2" in name:
            torch.nn.init.zeros_(parameter)


def dummy_batch(batch_size: int, seq_len: int, vocab_size: int, device: torch.device):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.roll(shifts=-1, dims=1)
    return input_ids, labels


def summarize_router(diag: RoutingDiagnostics, index: int) -> dict[str, float]:
    out = {
        f"router/{index}/entropy": float(diag.entropy.detach().cpu()),
        f"router/{index}/capacity_utilization": float(diag.capacity_utilization.detach().cpu()),
        f"router/{index}/matched_compute_fraction": float(
            diag.matched_compute_fraction.detach().cpu()
        ),
    }
    if diag.pressure is not None:
        out[f"router/{index}/pressure_mean"] = float(diag.pressure.float().mean().detach().cpu())
    if diag.z_loss is not None:
        out[f"router/{index}/z_loss"] = float(diag.z_loss.detach().cpu())
    return out


def run_case(
    *,
    name: str,
    router_cfg_path: str,
    model_cfg_path: str,
    init_mode: str,
    steps: int,
    batch_size: int,
    seq_len: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(seed)

    cfg = load_composed_cfg(router_cfg_path, model_cfg_path)
    model = DecoderOnlyLM(build_model_cfg(cfg)).to(device)
    if init_mode == "small":
        gpt_style_small_init(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    vocab_size = int(cfg.tokenizer.vocab_size)
    step_rows: list[dict[str, float]] = []

    model.train()
    for step in range(1, steps + 1):
        input_ids, labels = dummy_batch(batch_size, seq_len, vocab_size, device)
        optimizer.zero_grad(set_to_none=True)
        logits, total_loss, parts = model(input_ids, labels)
        if total_loss is None:
            raise RuntimeError("Model did not return a loss.")
        lm_loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )
        total_loss.backward()
        optimizer.step()

        row = {
            "step": step,
            "total_loss": float(total_loss.detach().cpu()),
            "lm_loss": float(lm_loss.detach().cpu()),
            "aux_loss": float(parts["aux_loss"].detach().cpu()),
            "logits_std": float(logits.detach().std().cpu()),
            "logits_abs_max": float(logits.detach().abs().max().cpu()),
        }
        for i, diag in enumerate(model.routing_diagnostics()):
            row.update(summarize_router(diag, i))
        step_rows.append(row)

    return {
        "case": name,
        "init_mode": init_mode,
        "seed": seed,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "expected_uniform_ce": float(torch.log(torch.tensor(vocab_size, dtype=torch.float32))),
        "steps": step_rows,
    }


def print_human_summary(results: list[dict[str, object]]) -> None:
    print("Startup loss reproduction on dummy byte-token data")
    print("Interpretation: if all routers start near the same huge LM loss, the issue is shared")
    print("model initialization/logit scale, not the reflected controller alone.")
    print()
    for result in results:
        first = result["steps"][0]
        last = result["steps"][-1]
        print(
            f"[{result['case']} | init={result['init_mode']}] "
            f"expected_ce~{result['expected_uniform_ce']:.3f} "
            f"step1_total={first['total_loss']:.3f} "
            f"step1_lm={first['lm_loss']:.3f} "
            f"step1_aux={first['aux_loss']:.3f} "
            f"step1_logits_std={first['logits_std']:.3f} "
            f"step{last['step']}_total={last['total_loss']:.3f}"
        )
    print()
    print(json.dumps(results, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the large startup loss and verify whether it comes from the "
            "shared model initialization or from a specific router architecture."
        )
    )
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--include-small-init",
        action="store_true",
        help="Also run the same cases after a small GPT-style reinitialization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    cases = [
        ("reflected_v2", "configs/router/reflected.yaml", "configs/model/tiny_moe.yaml"),
        ("top2", "configs/router/top2.yaml", "configs/model/tiny_moe.yaml"),
        ("dense", "configs/router/top1.yaml", "configs/model/tiny_dense.yaml"),
    ]

    init_modes = ["default"]
    if args.include_small_init:
        init_modes.append("small")

    results: list[dict[str, object]] = []
    for case_name, router_cfg_path, model_cfg_path in cases:
        for init_mode in init_modes:
            results.append(
                run_case(
                    name=case_name,
                    router_cfg_path=router_cfg_path,
                    model_cfg_path=model_cfg_path,
                    init_mode=init_mode,
                    steps=args.steps,
                    batch_size=args.batch_size,
                    seq_len=args.seq_len,
                    lr=args.lr,
                    seed=args.seed,
                    device=device,
                )
            )

    print_human_summary(results)


if __name__ == "__main__":
    main()
