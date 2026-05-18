# moe-route

Research code for load-aware reflected routing in decoder-only Mixture-of-Experts language models.

The repository uses a Python 3.11 `src/` layout, Hydra configs, plain PyTorch training, DDP via `torchrun`, and MLflow-compatible tracking.

## Safe local checks

```powershell
uv sync --extra dev
uv run pytest
uv run ruff check .
```

The default configs are intentionally small and CPU-friendly for smoke validation. Main FineWeb experiments are configured but should be launched only on a Linux GPU environment with prepared cache/output paths.

## Entrypoints

```powershell
python -m moe_route.cli.train experiment=smoke_reflected
python -m moe_route.cli.eval_ppl checkpoint=artifacts/checkpoints/latest.pt
python -m moe_route.cli.eval_tasks checkpoint=artifacts/checkpoints/latest.pt
python -m moe_route.cli.analyze_routing run=artifacts/runs/example
python -m moe_route.cli.sweep experiment=smoke_reflected --multirun
```

