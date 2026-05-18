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
python -m moe_route.cli.prepare_data data=tinystories_smoke
python -m moe_route.cli.eval_ppl checkpoint=artifacts/checkpoints/latest.pt
python -m moe_route.cli.eval_tasks checkpoint=artifacts/checkpoints/latest.pt
python -m moe_route.cli.analyze_routing run=artifacts/runs/example
python -m moe_route.cli.sweep experiment=smoke_reflected --multirun
```

## Runner Scripts

The `scripts/` launchers discover available Hydra config groups and expose typed choices:

```powershell
python scripts/prepare_data.py --data tinystories_smoke --rebuild
python scripts/launch_train.py --experiment smoke_reflected --data tinystories_smoke --router reflected_top2 --model tiny_moe --set trainer.max_steps=100
python scripts/prepare_data.py --data tinystories
python scripts/launch_train.py --experiment tinystories_reflected --data tinystories --trainer tinystories --router reflected_top2 --model tiny_moe
python scripts/launch_tinystories_suite.py --epochs 5
python scripts/launch_train.py --experiment fineweb_reflected_10bt --data fineweb_10bt --router reflected_top2 --nproc-per-node 8 --set trainer.precision=bf16
```

Training automatically validates or prepares packed token shards when `trainer.prepare_data=true`.
