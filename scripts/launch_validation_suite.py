from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _runner import module_cmd, run_command

SUITE = ["dense", "top1", "top2", "reflected"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run validation suite across all MoE experiments.")
    parser.add_argument(
        "--only",
        choices=SUITE,
        action="append",
        default=[],
        help="Run only selected experiment(s). Can be provided multiple times.",
    )
    parser.add_argument("--run-prefix", default="tinystories")
    parser.add_argument("--checkpoint-root", default="artifacts/checkpoints/tinystories")
    parser.add_argument("--output-dir", default="artifacts/validation_results", help="Directory to save JSON/CSV validation data.")
    parser.add_argument("--eval-ppl", action="store_true", help="Run perplexity evaluation.")
    parser.add_argument("--eval-tasks", action="store_true", help="Run downstream tasks evaluation.")
    parser.add_argument("--analyze-routing", action="store_true", help="Analyze routing metrics (MoE only).")
    parser.add_argument("--all-checkpoints", action="store_true", help="Evaluate all checkpoints instead of just the latest.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = [s for s in SUITE if not args.only or s in args.only]

    if not (args.eval_ppl or args.eval_tasks or args.analyze_routing):
        print("WARNING: No evaluation type selected. Please specify --eval-ppl, --eval-tasks, and/or --analyze-routing.")
        return

    root = Path(args.checkpoint_root)

    for spec in selected:
        run_name = f"{args.run_prefix}_{spec}"
        run_dir = root / run_name

        if not run_dir.exists():
            print(f"\n[SKIP] {run_name}: Directory {run_dir} does not exist.")
            continue

        out_dir = Path(args.output_dir) / run_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}\n[EVALUATING]: {run_name}\n{'='*60}")

        # 1. Routing Analysis (Only for MoE models)
        if args.analyze_routing and spec != "dense":
            print(f"\n--- Analyzing Routing Metrics for {run_name} ---")
            routing_out = out_dir / "routing_metrics.json"
            cmd = module_cmd("moe_route.cli.analyze_routing", [f"+run={run_dir}", f"+output_file={routing_out}"])
            run_command(cmd, args.dry_run)

        # 2. PPL and Task Evaluation
        if args.eval_ppl or args.eval_tasks:
            checkpoints = list(run_dir.glob("step_*.pt"))
            if not checkpoints:
                print(f"No checkpoints found in {run_dir}.")
                continue
            
            # Sort checkpoints numerically by step
            def get_step(p: Path) -> int:
                try:
                    return int(p.stem.split("_")[-1])
                except ValueError:
                    return -1
                    
            checkpoints.sort(key=get_step)
            to_evaluate = checkpoints if args.all_checkpoints else [checkpoints[-1]]

            for ckpt in to_evaluate:
                print(f"\n--- Evaluating Checkpoint: {ckpt.name} ---")
                if args.eval_ppl:
                    ppl_out = out_dir / f"ppl_{ckpt.stem}.json"
                    cmd = module_cmd("moe_route.cli.eval_ppl", [f"+checkpoint={ckpt}", f"+output_file={ppl_out}"])
                    run_command(cmd, args.dry_run)
                if args.eval_tasks:
                    tasks_out = out_dir / f"tasks_{ckpt.stem}.json"
                    cmd = module_cmd("moe_route.cli.eval_tasks", [f"+checkpoint={ckpt}", f"+output_file={tasks_out}"])
                    run_command(cmd, args.dry_run)


if __name__ == "__main__":
    main()
