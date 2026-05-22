"""Analyze and compare reflected_sparse vs top2 training runs."""
import json
import csv
import math
from pathlib import Path
from collections import OrderedDict

BASE = Path(r"C:\Users\Hellx\Documents\Programming\python\Project\Neryva\moe_route\training_logs")

# ---------------------------------------------------------------------------
# 1. Parameters
# ---------------------------------------------------------------------------
def load_params(path):
    with open(path) as f:
        return json.load(f)

ref_params = load_params(BASE / "reflected" / "artifacts" / "runs" / "tinystories_reflected_sparse" / "params.json")
top2_params = load_params(BASE / "top2" / "artifacts" / "runs" / "tinystories_top2" / "params.json")

def diff_params(r, t, key):
    a = r.get(key, {})
    b = t.get(key, {})
    if isinstance(a, dict) and isinstance(b, dict):
        return {k: (a.get(k), b.get(k)) for k in set(list(a.keys())+list(b.keys())) if a.get(k) != b.get(k)}
    if a != b:
        return (a, b)
    return None

print("=" * 90)
print("CONFIGURATION DIFFERENCES")
print("=" * 90)
print(f"{'Parameter':<35} {'Reflected Sparse':<25} {'Top2':<25}")
print("-" * 90)

# Model
print(f"\n--- Model ---")
for k in ["num_experts", "expert_hidden_size", "n_layers", "n_heads", "d_model", "d_ff"]:
    rv = ref_params.get("model", {}).get("moe" if k in ["num_experts","expert_hidden_size"] else "", ref_params.get("model", {}).get(k, "N/A"))
    tv = top2_params.get("model", {}).get("moe" if k in ["num_experts","expert_hidden_size"] else "", top2_params.get("model", {}).get(k, "N/A"))
    if k in ["num_experts","expert_hidden_size"]:
        rv = ref_params["model"]["moe"].get(k)
        tv = top2_params["model"]["moe"].get(k)
    else:
        rv = ref_params["model"].get(k)
        tv = top2_params["model"].get(k)
    if rv != tv:
        print(f"{'moe.'+k:<35} {str(rv):<25} {str(tv):<25}")

# Router
print(f"\n--- Router ---")
for k in ["name", "kind", "top_k", "capacity_factor", "drop_tokens"]:
    rv = ref_params.get("router", {}).get(k, "N/A")
    tv = top2_params.get("router", {}).get(k, "N/A")
    if rv != tv:
        print(f"{'router.'+k:<35} {str(rv):<25} {str(tv):<25}")
# Extra router params for reflected
for k in ["aux_loss_weight", "z_loss_weight", "temperature", "pressure_scale", "pressure_lr", "learnable_bias"]:
    rv = ref_params.get("router", {}).get(k, "N/A")
    tv = top2_params.get("router", {}).get(k, "N/A")
    if rv != tv:
        print(f"{'router.'+k:<35} {str(rv):<25} {str(tv):<25}")

# Data
print(f"\n--- Data ---")
for k in ["batch_size", "sequence_length", "streaming"]:
    rv = ref_params.get("data", {}).get(k, "N/A")
    tv = top2_params.get("data", {}).get(k, "N/A")
    if rv != tv:
        print(f"{'data.'+k:<35} {str(rv):<25} {str(tv):<25}")

# Trainer
print(f"\n--- Trainer ---")
for k in ["max_steps", "precision", "log_every", "checkpoint_every"]:
    rv = ref_params.get("trainer", {}).get(k, "N/A")
    tv = top2_params.get("trainer", {}).get(k, "N/A")
    if rv != tv:
        print(f"{'trainer.'+k:<35} {str(rv):<25} {str(tv):<25}")

# ---------------------------------------------------------------------------
# 2. Validation Metrics (CSV) - key checkpoints side-by-side
# ---------------------------------------------------------------------------
def read_validation_csv(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

ref_val = read_validation_csv(BASE / "reflected" / "artifacts" / "validation_results" / "tinystories_reflected_sparse" / "metrics.csv")
top2_val = read_validation_csv(BASE / "top2" / "artifacts" / "validation_results" / "tinystories_top2" / "metrics.csv")

print("\n" + "=" * 90)
print("VALIDATION METRICS - SIDE-BY-SIDE BY CHECKPOINT STEP")
print("=" * 90)

KEY_METRICS = [
    ("eval/loss", "Loss"),
    ("eval/lm_loss", "LM Loss"),
    ("eval/ppl", "PPL"),
    ("eval/bpb", "BPB"),
    ("eval/aux_loss", "Aux Loss"),
    ("eval/top1_acc", "Top-1 Acc"),
    ("eval/top5_acc", "Top-5 Acc"),
    ("eval/early_to_late_ratio", "Early-Late Ratio"),
    ("eval/pos_early_loss", "Pos-Early Loss"),
    ("eval/pos_mid_early_loss", "Pos-MidEarly Loss"),
    ("eval/pos_middle_loss", "Pos-Middle Loss"),
    ("eval/pos_mid_late_loss", "Pos-MidLate Loss"),
    ("eval/pos_late_loss", "Pos-Late Loss"),
    ("eval/domain_normal_loss", "Domain Normal Loss"),
    ("eval/domain_normal_ppl", "Domain Normal PPL"),
    ("eval/domain_dialogue_loss", "Domain Dialogue Loss"),
    ("eval/domain_dialogue_ppl", "Domain Dialogue PPL"),
    ("eval/domain_punctuation_loss", "Domain Punct Loss"),
    ("eval/domain_punctuation_ppl", "Domain Punct PPL"),
    ("eval/domain_ppl_spread", "Domain PPL Spread"),
    ("eval/domain_ppl_std", "Domain PPL StdDev"),
    ("eval/max_sample_loss", "Max Sample Loss"),
    ("eval/p99_loss", "P99 Loss"),
    ("eval/p95_loss", "P95 Loss"),
    ("eval/p90_loss", "P90 Loss"),
    ("eval/median_loss", "Median Loss"),
    ("eval/min_sample_loss", "Min Sample Loss"),
    ("eval/p95_to_median_ratio", "P95/Median Ratio"),
]

# Build lookup by step
ref_by_step = {r["step"]: r for r in ref_val}
top2_by_step = {r["step"]: r for r in top2_val}
common_steps = sorted(set(ref_by_step.keys()) & set(top2_by_step.keys()), key=int)

print(f"\n{'Step':<6} {'Metric':<28} {'Reflected Sparse':<18} {'Top2':<18} {'Delta (Ref-Top2)':<18} {'Better?':<10}")
print("-" * 100)

for step in common_steps:
    r = ref_by_step[step]
    t = top2_by_step[step]
    print(f"\n--- Step {step} ---")
    for col, label in KEY_METRICS:
        rv = float(r.get(col, 0))
        tv = float(t.get(col, 0))
        delta = rv - tv
        # Determine which is better (lower is better for loss/ppl, higher for acc)
        if "acc" in col:
            better = "Reflected" if rv > tv else "Top2" if tv > rv else "Tie"
        elif "aux_loss" in col:
            better = "Reflected" if abs(rv) < abs(tv) else "Top2" if abs(tv) < abs(rv) else "Tie"
        else:
            better = "Reflected" if rv < tv else "Top2" if tv < rv else "Tie"
        if rv != tv:
            print(f"{'':<6} {label:<28} {rv:<18.6f} {tv:<18.6f} {delta:<+18.6f} {better:<10}")

print("\n" + "=" * 90)
print("FINAL CHECKPOINT (step=76000) - SUMMARY COMPARISON")
print("=" * 90)
r = ref_by_step["76000"]
t = top2_by_step["76000"]
print(f"\n{'Metric':<30} {'Reflected Sparse':<20} {'Top2':<20} {'Delta (Ref-Top2)':<20}")
print("-" * 90)
final_metrics = [
    ("eval/loss", "Loss"),
    ("eval/lm_loss", "LM Loss"),
    ("eval/ppl", "Perplexity"),
    ("eval/bpb", "BPB"),
    ("eval/aux_loss", "Aux Loss"),
    ("eval/top1_acc", "Top-1 Accuracy"),
    ("eval/top5_acc", "Top-5 Accuracy"),
    ("eval/early_to_late_ratio", "Early-Late Ratio"),
    ("eval/pos_early_loss", "Position-Early Loss"),
    ("eval/pos_late_loss", "Position-Late Loss"),
    ("eval/domain_normal_loss", "Domain-Normal Loss"),
    ("eval/domain_normal_ppl", "Domain-Normal PPL"),
    ("eval/domain_dialogue_loss", "Domain-Dialogue Loss"),
    ("eval/domain_dialogue_ppl", "Domain-Dialogue PPL"),
    ("eval/domain_punctuation_loss", "Domain-Punct Loss"),
    ("eval/domain_punctuation_ppl", "Domain-Punct PPL"),
    ("eval/domain_ppl_spread", "Domain PPL Spread"),
    ("eval/domain_ppl_std", "Domain PPL StdDev"),
    ("eval/max_sample_loss", "Max Sample Loss"),
    ("eval/p99_loss", "P99 Loss"),
    ("eval/p95_loss", "P95 Loss"),
    ("eval/p90_loss", "P90 Loss"),
    ("eval/median_loss", "Median Loss"),
    ("eval/min_sample_loss", "Min Sample Loss"),
    ("eval/p95_to_median_ratio", "P95/Median Ratio"),
]
for col, label in final_metrics:
    rv = float(r.get(col, 0))
    tv = float(t.get(col, 0))
    delta = rv - tv
    if "acc" in col:
        marker = " [Ref-Wins]" if rv > tv else " [Top2-Wins]"
    elif "aux_loss" in col:
        marker = " [Ref-Wins]" if abs(rv) < abs(tv) else " [Top2-Wins]"
    else:
        marker = " [Ref-Wins]" if rv < tv else " [Top2-Wins]"
    print(f"{label:<30} {rv:<20.6f} {tv:<20.6f} {delta:<+20.6f}{marker}")

# ---------------------------------------------------------------------------
# 3. Training Trajectory Summary (from JSONL)
# ---------------------------------------------------------------------------
def read_training_jsonl(path):
    metrics = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                metrics.append(json.loads(line))
    return metrics

ref_train = read_training_jsonl(BASE / "reflected" / "artifacts" / "runs" / "tinystories_reflected_sparse" / "metrics.jsonl")
top2_train = read_training_jsonl(BASE / "top2" / "artifacts" / "runs" / "tinystories_top2" / "metrics.jsonl")

print("\n" + "=" * 90)
print("TRAINING METRICS COMPARISON (selected milestones)")
print("=" * 90)

# Find milestones: first, step ~500, step ~1000, step ~5000, step ~10000, last
def find_milestones(metrics, steps):
    by_step = {m["step"]: m for m in metrics}
    result = {}
    for s in steps:
        # Find closest
        available = sorted(by_step.keys())
        closest = min(available, key=lambda x: abs(x - s))
        result[s] = by_step[closest]
    return result

milestones = [1, 10, 100, 500, 1000, 2000, 5000, 10000, 20000, 30000, 50000, 76000]
ref_ms = find_milestones(ref_train, milestones)
top2_ms = find_milestones(top2_train, milestones)

print(f"\n{'Step':<8} {'Metric':<25} {'Reflected Sparse':<20} {'Top2':<20}")
print("-" * 75)
train_metrics = ["train/loss", "train/lm_loss", "train/aux_loss", "router/0/entropy", "router/0/load_cv", "router/0/load_gini", "router/0/drop_rate", "router/0/matched_compute_fraction", "router/1/load_cv", "router/1/load_gini", "perf/tokens_per_sec"]

for step in milestones:
    r = ref_ms.get(step)
    t = top2_ms.get(step)
    if r is None and t is None:
        continue
    print(f"\n--- Step ~{step} ---")
    for col in train_metrics:
        rv = r.get(col, "N/A") if r else "N/A"
        tv = t.get(col, "N/A") if t else "N/A"
        if rv != "N/A" and tv != "N/A":
            rv = float(rv)
            tv = float(tv)
            print(f"{'':<8} {col:<25} {rv:<20.6f} {tv:<20.6f}")

# ---------------------------------------------------------------------------
# 4. Routing Stress Evaluation
# ---------------------------------------------------------------------------
def load_routing_stress(path):
    with open(path) as f:
        return json.load(f)

ref_stress = load_routing_stress(BASE / "reflected" / "artifacts" / "routing_stress" / "tinystories_reflected_sparse.json")
top2_stress = load_routing_stress(BASE / "top2" / "artifacts" / "routing_stress" / "tinystories_top2.json")

print("\n" + "=" * 90)
print("ROUTING STRESS TEST - BEST CHECKPOINT (step 76000, eval on train split)")
print("=" * 90)

rs_metrics = OrderedDict([
    ("eval/loss", "Eval Loss"),
    ("eval/lm_loss", "Eval LM Loss"),
    ("eval/ppl", "Eval Perplexity"),
    ("eval/bpb", "Eval BPB"),
    ("eval/aux_loss", "Aux Loss"),
    ("router/0/drop_rate", "Router0 Drop Rate"),
    ("router/0/dropped_assignments", "Router0 Dropped Assign"),
    ("router/0/entropy", "Router0 Entropy"),
    ("router/0/matched_compute_fraction", "Router0 Matched Compute"),
    ("router/0/capacity_utilization", "Router0 Capacity Util"),
    ("router/0/load_cv", "Router0 Load CV"),
    ("router/0/load_gini", "Router0 Load Gini"),
    ("router/0/overflow", "Router0 Overflow"),
    ("router/0/pressure_mean", "Router0 Pressure Mean"),
    ("router/1/drop_rate", "Router1 Drop Rate"),
    ("router/1/dropped_assignments", "Router1 Dropped Assign"),
    ("router/1/entropy", "Router1 Entropy"),
    ("router/1/matched_compute_fraction", "Router1 Matched Compute"),
    ("router/1/capacity_utilization", "Router1 Capacity Util"),
    ("router/1/load_cv", "Router1 Load CV"),
    ("router/1/load_gini", "Router1 Load Gini"),
    ("router/1/overflow", "Router1 Overflow"),
    ("router/1/pressure_mean", "Router1 Pressure Mean"),
])

best_r = ref_stress["checkpoints"]["best"]
best_t = top2_stress["checkpoints"]["best"]

print(f"\n{'Metric':<35} {'Reflected Sparse':<22} {'Top2':<22} {'Delta':<15}")
print("-" * 95)
for col, label in rs_metrics.items():
    rv = best_r.get(col)
    tv = best_t.get(col)
    if rv is None or tv is None:
        continue
    if isinstance(rv, float) and isinstance(tv, float):
        if rv == float('inf') and tv == float('inf'):
            rv_str = "inf"
            tv_str = "inf"
            delta_str = "N/A"
        else:
            delta = rv - tv
            rv_str = f"{rv:<22.6f}" if abs(rv) < 1e10 else f"{rv:<22.4e}"
            tv_str = f"{tv:<22.6f}" if abs(tv) < 1e10 else f"{tv:<22.4e}"
            delta_str = f"{delta:<+15.6f}" if abs(delta) < 1e10 else f"{delta:<+15.4e}"
        print(f"{label:<35} {rv_str} {tv_str} {delta_str}")
    else:
        print(f"{label:<35} {str(rv):<22} {str(tv):<22} {'N/A':<15}")

# ---------------------------------------------------------------------------
# 5. Console log summary (training duration)
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("CONSOLE LOG TRAINING SUMMARY")
print("=" * 90)
# from console.md for reflected
print("\nReflected Sparse - trained ~9 epochs (stopped at ~77040 steps)")
print("  Final loss reported: ~0.864 at step 77040")
print("  Average throughput: ~785k tokens/sec")
print("  Router: drop_rate~0.40, compute_fraction~0.50, entropy~1.20, load_cv~0.30")

print("\nTop2 - trained 76000 steps (19 checkpoints)")
print("  (console log only shows data preparation, not training output)")

print("\n" + "=" * 90)
print("KEY FINDINGS SUMMARY")
print("=" * 90)
print("""
1. VALIDATION LOSS: Reflected Sparse achieves LOWER validation loss (0.7998 vs 0.8330 at step 76000)
2. PERPLEXITY: Reflected Sparse achieves LOWER perplexity (2.225 vs 2.300 at step 76000)
3. TOP-1 ACCURACY: Reflected Sparse is HIGHER (0.7501 vs 0.7397 at step 76000)
4. AUXILIARY LOSS: Reflected Sparse has ZERO aux loss (aux-loss-free by design), Top2 uses 0.01 weight
5. ROUTER LOAD BALANCING: 
   - Top2 has better load balance (lower CV, lower Gini) at Router0
   - But Reflected achieves this WITHOUT any auxiliary balancing loss
6. ROUTING STRESS: Both exhibit infinite loss under stress test (eval on train with different capacity)
7. CAPACITY UTILIZATION: Both are similar (~0.5 matched compute fraction)
8. ENTROPY: Reflected has much lower entropy (~1.08-1.18) vs Top2 (~2.50-2.58), 
   indicating more decisive/expert-specialized routing
9. DOMAIN PERFORMANCE: Reflected Sparse wins on all 3 domains (normal, dialogue, punctuation)
""")
