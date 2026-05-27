# Publication Plan for Reflected MoE v2

## Summary

Use the **current reflected implementation and its real training artifacts** as the source of truth, not the legacy hypothesis docs. Concretely, anchor the paper on the reflected sparse router recorded in [comparison_report.md](C:/Users/Hellx/Documents/Programming/python/Project/Neryva/moe_route/training_logs/comparison_report.md) and [resolved_config.yaml](C:/Users/Hellx/Documents/Programming/python/Project/Neryva/moe_route/training_logs/refleceted_v2/artifacts/runs/tinystories_reflected_sparse/resolved_config.yaml): `kind=reflected_v2`, `routing_mode=sparse`, `top_k=2`, `capacity_factor=0.5`, `shared_experts=1`, `pressure_lr=0.05`, `pressure_warmup_steps=1000`, `aux_loss=0`.

The current results are **promising but not publication-ready for a top-tier main track**. The blocking gaps are:
- evidence is still almost entirely **TinyStories / tiny model**;
- downstream evaluation is **not operational yet** for native checkpoints;
- the stress result is **not trustworthy enough to claim**, because all methods fail in nearly the same catastrophic way;
- there are **no multi-seed main results**;
- there is **no main-scale matched-compute result** yet for the current reflected method against **DeepSeek-LFB**, nor a **scale anchor comparison** against standard `Top-2`.

Primary target: **ICLR 2027 main track** as the first realistic conference cycle; this is an inference based on the fact that **ICML 2026 closed on January 28, 2026** and **NeurIPS 2026 closed on May 6, 2026**.  
Grounding sources: [ICML 2026 CFP](https://icml.cc/Conferences/2026/CallForPapers), [NeurIPS 2026 CFP](https://neurips.cc/Conferences/2026/CallForPapers), [NeurIPS checklist](https://neurips.cc/public/guides/PaperChecklist), [ST-MoE](https://arxiv.org/abs/2202.08906), [Loss-Free Balancing](https://arxiv.org/abs/2408.15664), [OpenMoE](https://arxiv.org/abs/2402.01739), [OLMoE](https://arxiv.org/abs/2409.02060).

## Key Changes

### 1. Freeze the paper method now
- Treat **Reflected v2** as the paper method.
- Treat **old reflected/V1** as historical only; use it only in appendix if needed to explain the V2 improvement.
- Do not change the algorithm again before the decisive baseline suite is complete, except for bug fixes proven not to alter semantics.

### 2. Close the tooling gaps before spending more training budget
- Add a **native checkpoint adapter/export path** so `lm-eval` can run on your checkpoints; without this, downstream-task claims are blocked.
- Add a **single experiment manifest** per run: config, seed, git SHA, token count, wall-clock time, throughput, checkpoint paths, eval outputs.
- Audit the **routing stress evaluation** and either:
  - fix it into a meaningful robustness benchmark, or
  - remove it from the main claim and keep it out of the paper until validated.

### 3. Run the experiment program in this order
- **Phase A: TinyStories baseline comparison**
  - Baselines: `Top-2`, `DeepSeek-LFB`, `Reflected v2`.
  - Supporting controls: `Dense` and `Top-1` on the same tiny setting only.
  - Use **identical non-router settings** across the three main methods.
  - Run the main TinyStories comparison at `capacity_factor in {0.5, 1.0}` to separate constrained-capacity behavior from less-stressed behavior.
  - Run **3 seeds** for the three main methods.
  - Required outputs: validation loss/PPL, top-1 accuracy, router CV/Gini, drop rate, entropy, throughput, early-vs-late loss, loss tails.

- **Phase B: small-scale ablations for the reflected mechanism only**
  - Use **TinyStories only** for the ablation program so FineWeb budget is reserved for the final baseline comparisons.
  - Sweep exactly these reflected knobs: `pressure_lr {0.01, 0.05, 0.1}`, `capacity_factor {0.5, 1.0}`, `shared_experts {0,1}`, `pressure_warmup_steps {0,1000}`, `learnable_bias {off,on}`.
  - Optional only if stable in code: `gate_function {softmax,sigmoid}`.
  - The ablation study varies **only reflected variants**; `Top-2` and `DeepSeek-LFB` are included only as **standard reference baselines** so each reflected variant can be judged against strong non-reflected alternatives.
  - Report the reference baselines in the same TinyStories tables/plots used for the ablation analysis, but do not treat those external routers as ablated method variants.
  - Keep all ablations on TinyStories; do not spend FineWeb budget on ablation breadth.

- **Phase C: final FineWeb baseline comparison**
  - Dataset: `fineweb_10bt` already defined in the repo.
  - Main FineWeb comparison: `Reflected v2` vs `DeepSeek-LFB`.
  - Keep **one Top-2 anchor run** on FineWeb at the same architecture and training setup so the paper still has a standard sparse-routing scale reference.
  - Official capacity policy is `capacity_factor in {0.5, 1.0}`; if FineWeb budget becomes tight, preserve both capacities for `Reflected v2` and `DeepSeek-LFB` first, and only then reduce the `Top-2` anchor coverage.
  - This is the **final multi-seed main-result section** of the project; FineWeb is not used for reflected ablation breadth.
  - Seed policy: run **multi-seed FineWeb experiments** for the main baselines, with **2 seeds** for `Reflected v2`, **2 seeds** for `DeepSeek-LFB`, and **1 seed** for the `Top-2` anchor as the minimum acceptable plan.
  - Evaluate held-out perplexity plus a compact zero-shot suite: **HellaSwag, PIQA, WinoGrande, ARC-Easy, ARC-Challenge**.
  - Report compute-normalized results: active parameters, tokens, wall-clock, tokens/sec, and routing overhead.

### 4. Make the paper claim narrower and stronger
- Main claim should be: **aux-loss-free reflected routing improves language-model quality and routing behavior under matched compute relative to Top-2 and DeepSeek-LFB**.
- Do **not** claim universal superiority, frontier-scale scalability, or robustness under distribution shift until the stress benchmark is fixed.
- Use `Dense` only as a supporting compute-quality control, not as the main comparison axis.

## Test Plan

- **Reproducibility**
  - same seed rerun reproduces metrics within a small tolerance;
  - resolved configs and manifests are complete for every reported run.

- **Evaluation integrity**
  - `lm-eval` works directly on your model artifacts;
  - stress benchmark differentiates sane/broken checkpoints, otherwise it is excluded.

- **Paper acceptance criteria**
  - TinyStories: reflected v2 beats Top-2 and DeepSeek-LFB on mean validation loss over 3 seeds;
  - Capacity study: reflected v2 shows its strongest advantage at `capacity_factor=0.5` and remains competitive at `capacity_factor=1.0`;
  - FineWeb: reflected v2 beats or matches DeepSeek-LFB on the main metrics, and remains competitive against the Top-2 anchor run;
  - throughput overhead is reported and remains practically competitive;
  - all main tables can be regenerated from scripts and saved manifests.

## Assumptions and Defaults

- Legacy hypothesis docs are **not authoritative** unless they match the current reflected code path.
- The winning method is the **sparse top-2 reflected v2** configuration in the training logs, not the smoke artifacts under `artifacts/runs`.
- `capacity_factor=0.5` is treated as a **deliberate constrained-capacity stress regime**, while `capacity_factor=1.0` is treated as the less-stressed comparison regime.
- Because your compute is **single/few-GPU moderate**, the correct strategy is **one strong main-scale comparison plus broad small-scale ablations**, not a wide multi-scale sweep.
- Since **ICML 2026** and **NeurIPS 2026** are already closed, the working submission posture is **prepare for the next announced major cycle, with ICLR 2027 as the first practical target**.
