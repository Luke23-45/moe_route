import json

def extract_detailed_stats(filepath):
    steps = [10, 1000, 10000, 50000, 100000, 150000, 193260]
    results = {}
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            step = data.get('step')
            if step in steps:
                results[step] = {
                    'loss': data.get('train/loss'),
                    'r0_ent': data.get('router/0/entropy'),
                    'r1_ent': data.get('router/1/entropy'),
                    'r0_z': data.get('router/0/z_loss', 0),
                    'r1_z': data.get('router/1/z_loss', 0),
                    'r0_cap': data.get('router/0/capacity_utilization'),
                    'r1_cap': data.get('router/1/capacity_utilization'),
                    'r0_over': data.get('router/0/overflow', 0),
                    'r1_over': data.get('router/1/overflow', 0)
                }
    return results

ref = extract_detailed_stats('C:/Users/Hellx/Documents/Programming/python/Project/Neryva/moe_route/training_logs/reflected/artifacts/runs/tinystories_reflected/metrics.jsonl')
top = extract_detailed_stats('C:/Users/Hellx/Documents/Programming/python/Project/Neryva/moe_route/training_logs/top2/artifacts/runs/tinystories_top2/metrics.jsonl')

print("=== REFLECTED DETAILED METRICS ===")
for s in [10, 1000, 10000, 50000, 100000, 150000, 193260]:
    d = ref.get(s, {})
    print(f"Step {s:6d} | Loss: {d.get('loss'):.4f} | R0 Ent: {d.get('r0_ent'):.4f} | R1 Ent: {d.get('r1_ent'):.4f} | R0 Z: {d.get('r0_z'):.6f} | R1 Z: {d.get('r1_z'):.6f} | R0 Cap: {d.get('r0_cap'):.2f} | R1 Cap: {d.get('r1_cap'):.2f} | R0 Over: {d.get('r0_over'):.1f} | R1 Over: {d.get('r1_over'):.1f}")

print("\n=== STANDARD TOP2 DETAILED METRICS ===")
for s in [10, 1000, 10000, 50000, 100000, 150000, 193260]:
    d = top.get(s, {})
    print(f"Step {s:6d} | Loss: {d.get('loss'):.4f} | R0 Ent: {d.get('r0_ent'):.4f} | R1 Ent: {d.get('r1_ent'):.4f} | R0 Z: {d.get('r0_z'):.6f} | R1 Z: {d.get('r1_z'):.6f} | R0 Cap: {d.get('r0_cap'):.2f} | R1 Cap: {d.get('r1_cap'):.2f} | R0 Over: {d.get('r0_over'):.1f} | R1 Over: {d.get('r1_over'):.1f}")
