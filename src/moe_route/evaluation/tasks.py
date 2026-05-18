from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def evaluate_tasks(cfg, checkpoint: str) -> dict[str, float]:
    """Run an lm-evaluation-harness subprocess for configured task names.

    The adapter is intentionally external-process based because the harness API changes more often
    than its CLI. For this project's native checkpoints, export or wrap the model into a harness
    compatible model before selecting downstream tasks.
    """
    tasks = list(cfg.eval.tasks)
    if not tasks:
        return {}

    adapter = cfg.eval.task_adapter
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "results.json"
        
        model_args = str(adapter.model_args or "")
        if "{checkpoint}" in model_args:
            model_args = model_args.format(checkpoint=checkpoint)
        elif not model_args and str(adapter.model) == "hf":
            model_args = f"pretrained={checkpoint}"

        cmd = [
            sys.executable,
            "-m",
            "lm_eval",
            "--model",
            str(adapter.model),
            "--model_args",
            model_args,
            "--tasks",
            ",".join(tasks),
            "--batch_size",
            str(adapter.batch_size),
            "--output_path",
            str(output_path),
        ]
        if adapter.limit is not None:
            cmd.extend(["--limit", str(adapter.limit)])

        completed = subprocess.run(cmd, check=False, text=True, capture_output=True)
        if completed.returncode != 0:
            raise RuntimeError(
                "lm-evaluation-harness failed. "
                f"stderr:\n{completed.stderr.strip()}\nstdout:\n{completed.stdout.strip()}"
            )
        if not output_path.exists():
            return {}
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        results = payload.get("results", payload)
        flat: dict[str, float] = {}
        for task_name, metrics in results.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    if isinstance(value, int | float):
                        flat[f"tasks/{task_name}/{metric_name}"] = float(value)
        return flat
