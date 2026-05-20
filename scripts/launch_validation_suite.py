import argparse
import csv
import sys
from pathlib import Path

# Add src to PYTHONPATH so we can import moe_route natively
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from _runner import module_cmd, run_command

from moe_route.training.checkpoint import load_checkpoint
from moe_route.evaluation.ppl import build_eval_data_cfg, evaluate_model_perplexity, load_cfg_from_checkpoint
from moe_route.data.pipeline import build_dataloader
from moe_route.models.transformer import DecoderOnlyLM, build_model_cfg
from moe_route.tokenization.tokenizers import build_tokenizer
from moe_route.evaluation.tasks import evaluate_tasks
from moe_route.analysis.routing import summarize_routing_run

SUITE = ["dense", "top1", "top2", "reflected", "reflected_sparse"]

def get_step(p: Path) -> int:
    try:
        return int(p.stem.split("_")[-1])
    except ValueError:
        return -1

def main() -> None:
    parser = argparse.ArgumentParser(description="Run validation suite across all MoE experiments.")
    parser.add_argument("--only", choices=SUITE, action="append", default=[])
    parser.add_argument("--run-prefix", default="tinystories")
    parser.add_argument("--checkpoint-root", default="artifacts/checkpoints/tinystories")
    parser.add_argument("--output-dir", default="artifacts/validation_results")
    parser.add_argument("--eval-ppl", action="store_true")
    parser.add_argument("--eval-tasks", action="store_true")
    parser.add_argument("--analyze-routing", action="store_true")
    parser.add_argument("--all-checkpoints", action="store_true")
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

        # 1. Routing Analysis
        if args.analyze_routing and spec != "dense" and not args.dry_run:
            print(f"\n--- Analyzing Routing Metrics for {run_name} ---")
            try:
                routing_metrics = summarize_routing_run(run_dir)
                routing_out = out_dir / "routing_summary.json"
                import json
                with open(routing_out, "w") as f:
                    json.dump(routing_metrics, f, indent=2)
                print(f"Saved Routing analysis to {routing_out}")
            except Exception as e:
                print(f"Failed to analyze routing: {e}")

        # 2. PPL and Task Evaluation
        if args.eval_ppl or args.eval_tasks:
            checkpoints = list(run_dir.glob("step_*.pt"))
            if not checkpoints:
                print(f"No checkpoints found in {run_dir}.")
                continue
            
            checkpoints.sort(key=get_step)
            to_evaluate_pool = checkpoints if args.all_checkpoints else [checkpoints[-1]]

            csv_path = out_dir / "metrics.csv"
            existing_data = []
            evaluated_keys = set()

            if csv_path.exists():
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Attempt to parse row values as floats where possible
                        parsed_row = {}
                        for k, v in row.items():
                            try:
                                parsed_row[k] = float(v) if "." in v or "e" in v.lower() else int(v)
                            except ValueError:
                                parsed_row[k] = v
                        existing_data.append(parsed_row)
                        eval_split = str(parsed_row.get("eval_split", ""))
                        evaluated_keys.add((int(parsed_row["step"]), eval_split))

            # --- One-Time In-Memory Setup ---
            print(f"Loading environment for {run_name}...")
            first_ckpt = to_evaluate_pool[0]
            eval_cfg = load_cfg_from_checkpoint(first_ckpt)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            eval_data_cfg = build_eval_data_cfg(eval_cfg)
            eval_split = str(eval_data_cfg.get("split", ""))
            
            model = None
            dataloader = None
            if args.eval_ppl:
                tokenizer = build_tokenizer(eval_cfg.tokenizer)
                dataloader = build_dataloader(eval_data_cfg, tokenizer)
                model = DecoderOnlyLM(build_model_cfg(eval_cfg)).to(device)
                pad_token_id = tokenizer.pad_token_id
                eval_precision = str(eval_cfg.trainer.get("precision", "fp32"))

            pending_checkpoints = [
                ckpt for ckpt in to_evaluate_pool if (get_step(ckpt), eval_split) not in evaluated_keys
            ]

            if not pending_checkpoints:
                print(f"All selected checkpoints for {run_name} have already been evaluated. Skipping.")
                continue

            if args.dry_run:
                print(f"[DRY-RUN] Would evaluate {len(pending_checkpoints)} checkpoints on split '{eval_split}'.")
                continue

            for ckpt in pending_checkpoints:
                step = get_step(ckpt)
                print(f"\n--- Evaluating Checkpoint: {ckpt.name} (Step {step}) ---")
                
                row = {"step": step, "eval_split": eval_split}
                
                if args.eval_ppl:
                    print("Running Perplexity Evaluation...")
                    model.to(device)
                    load_checkpoint(ckpt, model)
                    max_batches = int(eval_cfg.eval.max_batches)
                    ppl_metrics = evaluate_model_perplexity(
                        model, dataloader, max_batches, device,
                        ignore_index=pad_token_id,
                        precision=eval_precision,
                        tokenizer=tokenizer,
                    )
                    row.update(ppl_metrics)
                    print(f"PPL: {ppl_metrics.get('eval/ppl', 'N/A'):.2f}")
                
                if args.eval_tasks:
                    if args.eval_ppl:
                        # VRAM Management: Move model off GPU before launching lm-eval subprocess
                        model.cpu()
                        torch.cuda.empty_cache()

                    print("Running Downstream Tasks Evaluation...")
                    tasks_metrics = evaluate_tasks(eval_cfg, str(ckpt))
                    row.update(tasks_metrics)
                    print(f"Tasks evaluated: {len(tasks_metrics)} metrics collected.")

                existing_data.append(row)

                # Dynamically rewrite CSV with all keys
                all_keys = set()
                for r in existing_data:
                    all_keys.update(r.keys())
                
                headers = ["step"] + sorted([k for k in all_keys if k != "step"])
                
                temp_csv_path = csv_path.with_suffix(".csv.tmp")
                with open(temp_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(existing_data)
                
                # Atomic replacement
                temp_csv_path.replace(csv_path)
                
                print(f"Appended Step {step} results to {csv_path}")

if __name__ == "__main__":
    main()
