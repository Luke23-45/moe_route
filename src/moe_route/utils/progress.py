from __future__ import annotations

from dataclasses import dataclass

from tqdm.auto import tqdm


@dataclass(frozen=True)
class ProgressConfig:
    enabled: bool = True
    dynamic_ncols: bool = True
    leave: bool = True
    mininterval: float = 0.5


def progress_cfg(cfg) -> ProgressConfig:
    node = cfg.trainer.get("progress", {})
    return ProgressConfig(
        enabled=bool(node.get("enabled", True)),
        dynamic_ncols=bool(node.get("dynamic_ncols", True)),
        leave=bool(node.get("leave", True)),
        mininterval=float(node.get("mininterval", 0.5)),
    )


def training_bar(cfg, epoch: int, total: int, initial: int = 0):
    progress = progress_cfg(cfg)
    router_name = "dense" if not bool(cfg.model.moe.enabled) else str(cfg.router.name)
    desc = (
        f"{cfg.experiment.name} | data={cfg.data.name} | model={cfg.model.name} "
        f"| router={router_name} | epoch={epoch}"
    )
    return tqdm(
        total=total,
        initial=initial,
        desc=desc,
        unit="step",
        dynamic_ncols=progress.dynamic_ncols,
        leave=progress.leave,
        mininterval=progress.mininterval,
        disable=not progress.enabled,
    )
