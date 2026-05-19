from __future__ import annotations

import importlib.util
from pathlib import Path


def test_runner_discovers_config_choices() -> None:
    runner_path = Path(__file__).resolve().parents[2] / "scripts" / "_runner.py"
    spec = importlib.util.spec_from_file_location("_runner", runner_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "smoke_reflected" in module.config_choices("experiment")
    assert "tinystories_reflected" not in module.config_choices("experiment")
    assert "tinystories_smoke" in module.config_choices("data")
    assert "tinystories" in module.config_choices("data")
    assert "tinystories" in module.config_choices("trainer")
    assert "reflected_top2" not in module.config_choices("router")


def test_tinystories_suite_script_exists() -> None:
    path = Path(__file__).resolve().parents[2] / "scripts" / "launch_tinystories_suite.py"
    assert path.exists()


def test_routing_stress_suite_script_exists() -> None:
    path = Path(__file__).resolve().parents[2] / "scripts" / "launch_routing_stress_eval.py"
    assert path.exists()
