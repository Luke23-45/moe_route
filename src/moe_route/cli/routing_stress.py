from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig

from moe_route.evaluation.routing_stress import (
    evaluate_checkpoint_routing_stress,
    select_checkpoints,
    write_routing_stress_report,
)


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    checkpoint_dir = cfg.get("checkpoint_dir")
    if not checkpoint_dir:
        raise ValueError("checkpoint_dir=... is required.")

    validation_metrics_csv = cfg.get("validation_metrics_csv")
    eval_split = cfg.get("eval_split")
    max_batches = cfg.get("max_batches")
    selected = select_checkpoints(
        checkpoint_dir,
        validation_metrics_csv,
        eval_split=None if eval_split is None else str(eval_split),
    )

    results = {}
    for label, checkpoint in selected.items():
        if label == "latest" and selected.get("best") == checkpoint:
            continue
        results[label] = evaluate_checkpoint_routing_stress(
            checkpoint,
            max_batches=None if max_batches is None else int(max_batches),
        )
    if "best" not in results and "best" in selected:
        results["best"] = evaluate_checkpoint_routing_stress(
            selected["best"],
            max_batches=None if max_batches is None else int(max_batches),
        )

    any_metrics = next(iter(results.values()))
    payload = {
        "run_name": Path(checkpoint_dir).name,
        "eval_split": any_metrics.get("eval_split", ""),
        "prepared_split": any_metrics.get("prepared_split", ""),
        "checkpoints": results,
    }

    output_file = cfg.get("output_file")
    output_md = cfg.get("output_md")
    if output_file:
        write_routing_stress_report(output_file, payload, output_md=output_md)
        print(f"Saved routing stress report to {output_file}")
        if output_md:
            print(f"Saved routing stress markdown to {output_md}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
