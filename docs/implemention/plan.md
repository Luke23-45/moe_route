# Implementation Plan for the Load-Aware Reflected MoE Experiment Stack

## Summary

Build the project as a **Python 3.11, `uv`-managed, `pyproject.toml`-based, config-driven research codebase** with a **`src/` layout**, using:

- [`uv`](https://docs.astral.sh/uv/) for Python version management, dependency management, and lockfiles.
- [`pyproject.toml`](https://packaging.python.org/guides/writing-pyproject-toml/) as the single project and tooling configuration root.
- [`PyTorch`](https://pytorch.org/get-started/) as the core training framework.
- [`torch.distributed` + `torchrun` / DDP](https://docs.pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html) as the default distributed execution path.
- [`Hydra`](https://hydra.cc/docs/intro/) for hierarchical experiment configs and sweeps.
- [`datasets`](https://huggingface.co/docs/datasets/en/index) + [`tokenizers`](https://huggingface.co/docs/tokenizers/en/index) for corpus access, preprocessing, and tokenizer workflows.
- [`MLflow`](https://mlflow.org/docs/latest/ml/tracking/) as the default experiment tracker, with optional TensorBoard mirroring via [`torch.utils.tensorboard`](https://docs.pytorch.org/docs/stable/tensorboard.html).
- [`pytest`](https://docs.pytest.org/en/stable/explanation/goodpractices.html) + [`Ruff`](https://docs.astral.sh/ruff/) for testing and code quality.

Do **not** make Lightning, Accelerate, or DVC core dependencies in v1. They are useful tools, but this project depends on custom routing state, matched-compute accounting, and router diagnostics; plain PyTorch keeps the control surface tighter and easier to audit.

Use **Linux GPU cloud training as the primary execution target**, with Windows limited to local editing and CPU smoke tests.

## Key Changes

### Project architecture

Adopt this repo shape from day one:

```text
moe_route/
|- pyproject.toml
|- uv.lock
|- .python-version
|- README.md
|- docs/
|  |- plan/
|  |- implemention/
|  `- refs/
|- configs/
|  |- config.yaml
|  |- data/
|  |- tokenizer/
|  |- model/
|  |- router/
|  |- optimizer/
|  |- trainer/
|  |- distributed/
|  |- tracking/
|  |- eval/
|  `- experiment/
|- src/
|  `- moe_route/
|     |- cli/
|     |- config/
|     |- data/
|     |- tokenization/
|     |- models/
|     |- routing/
|     |- training/
|     |- evaluation/
|     |- tracking/
|     |- analysis/
|     `- utils/
|- tests/
|  |- unit/
|  |- integration/
|  `- smoke/
|- scripts/
|  |- launch_train.py
|  |- launch_eval.py
|  `- launch_sweep.py
`- artifacts/   # gitignored: checkpoints, outputs, mlruns, cache, reports
```

### Public interfaces and implementation contracts

Expose these stable entrypoints:

- `python -m moe_route.cli.train experiment=...`
- `python -m moe_route.cli.eval_ppl checkpoint=...`
- `python -m moe_route.cli.eval_tasks checkpoint=...`
- `python -m moe_route.cli.analyze_routing run=...`
- `python -m moe_route.cli.sweep experiment=... --multirun`
- `python scripts/launch_train.py ...`
- `python scripts/launch_eval.py ...`
- `python scripts/launch_sweep.py ...`

Define these internal interfaces explicitly before coding:

- `Router`: baseline and reflected routers share one forward contract returning expert assignments, combine weights, capacity statistics, and auxiliary diagnostics.
- `PressureState`: owns `init`, `update`, `project/reflection`, `serialize`, and `reset` behavior.
- `CapacityPolicy`: encapsulates top-k capacity, token dropping, overflow handling, and matched-compute accounting.
- `ExperimentTracker`: adapter interface with MLflow as the default implementation.
- `CorpusAdapter`: loads or streams a dataset split and yields normalized text or token samples.
- `DataPipeline`: owns tokenization mode, shard caching, sequence packing, batching, worker setup, prefetch behavior, and dataloader performance policy.
- `Evaluator`: separate interfaces for perplexity, downstream tasks, and routing diagnostics.

### System design choices

- Use a **code-first, config-driven** architecture: configs choose data, model, router, trainer, and evaluation variants; Python implements behavior.
- Keep the training loop in **plain PyTorch**, not Lightning.
- Use **DDP** as the only first-class multi-GPU path; do not support `DataParallel`.
- Make `torch.compile` optional behind config flags only after baseline correctness is proven.
- Default tracker is **MLflow local or SQLite-backed**; also emit structured JSON and CSV summaries so results are never trapped inside one UI.
- Keep data out of the repo; store dataset manifests, sampling configs, cache paths, and metadata only.
- Do **not** use the full FineWeb dataset in the default research path; the project should use staged data scales so implementation, debugging, and baseline comparison do not waste cloud compute.
- Make the first runnable path a **CPU smoke experiment** on TinyStories or another tiny local corpus; make the first serious path a **small-scale Linux GPU pretraining run** on an official sampled FineWeb variant.
- Replace shell launchers with **cross-platform Python launcher files** that validate configs, prepare output directories, and dispatch train, eval, and sweep jobs consistently across Windows-local and Linux-cloud environments.
- Optimize the pipeline end to end: keep preprocessing outside the hot training loop, prefer cached tokenized shards for main experiments, use packed fixed-length sequences, pinned-memory DataLoaders for CUDA runs, persistent workers when they help, and interval-based diagnostics so routing analysis does not dominate step time.
- Treat newer performance features as opt-in config flags: enable mixed precision, fused optimizers, compile mode, and dataloader parallelism only after correctness, reproducibility, and metric parity are verified.

## Implementation Sequence

1. **Foundation**
- Create the `uv` project, pin Python 3.11, and use `src/` layout.
- Configure `pyproject.toml` for runtime dependencies, dev dependencies, Ruff, pytest, and package entrypoints.
- Add repo policies: `.gitignore` for `artifacts/`, caches, checkpoints, `mlruns/`, and Hydra outputs.

2. **Config and CLI skeleton**
- Create Hydra root config and the config groups listed above.
- Implement `train`, `eval_ppl`, `eval_tasks`, `analyze_routing`, and `sweep` CLIs.
- Make every run write resolved config, seed, git commit, environment info, and artifact paths.
- Add Python launchers such as `scripts/launch_train.py`, `scripts/launch_eval.py`, and `scripts/launch_sweep.py` as the orchestration layer above the CLI entrypoints so the workflow stays cross-platform and argument-validated.

3. **Data and tokenization**
- Implement `CorpusAdapter` for TinyStories-smoke and FineWeb-main.
- Lock the main web-scale dataset to `HuggingFaceFW/fineweb` with `name="sample-10BT"` and `split="train"` as the default realistic pretraining corpus.
- Treat `sample-100BT` as an optional late-stage stronger-validation dataset only if the earlier reflected-router results justify the additional compute.
- Exclude full FineWeb from the default implementation and experiment path unless a later revision explicitly approves it.
- Support both pretrained tokenizer loading and custom BPE training via `tokenizers`.
- Add deterministic split and sequence-packing logic plus cacheable tokenized shards.
- Default the main experiment path to **pretokenized cached shards plus packed sequences** rather than raw-text-per-step processing.
- Keep a streaming mode available for large validation runs, but do not make streaming the default for the main matched-compute experiments because cached shards are easier to resume, benchmark, and compare fairly.
- Make dataloader settings explicit in config: worker count, prefetch factor, pinned memory, persistent workers, batch formation mode, and shuffle or shard policy.

4. **Model baselines**
- Implement the decoder-only dense baseline.
- Implement the MoE FFN insertion path with a shared backbone.
- Implement baseline routers first: dense, top-1, and top-2.

5. **Reflected router stack**
- Add the pressure-state module, load penalty term, reflection or projection step, and routing diagnostics.
- Support both soft training routing and top-k inference routing via config.
- Log pressure trajectories, routed mass, overload events, token drop, entropy, and churn inputs.

6. **Training, tracking, and evaluation**
- Add optimizer, scheduler, checkpoint, resume, and distributed modules.
- Add MLflow tracking with metric and artifact schema fixed up front.
- Add perplexity evaluation, routing-analysis reports, and an adapter for [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness) for downstream tasks.
- Include performance-critical trainer settings in config from the start: precision mode, gradient accumulation, gradient clipping, fused optimizer toggle, compile toggle, logging cadence, evaluation cadence, and checkpoint cadence.
- Design checkpoints for efficient cloud resumption: save model, optimizer, scheduler, pressure state, RNG state, resolved config, and sampler progress when supported.
- Keep metrics logging structured and sparse enough that instrumentation does not become the throughput bottleneck.

7. **Experiment matrix**
- Stage 1: smoke runs on TinyStories or another tiny local corpus.
- Stage 2: dense vs top-1 vs top-2 baselines.
- Stage 3: reflected-router main comparison on `HuggingFaceFW/fineweb`, `sample-10BT`.
- Stage 4: ablations for the pressure term, projection or barrier, expert count, layer placement, and soft-vs-hard routing.
- Stage 5: optional stronger-validation rerun on `sample-100BT` only if Stage 3 and Stage 4 show credible gains and the compute budget is approved.

## Test Plan

- **Unit tests**
- pressure update and reflection keep state valid and nonnegative.
- routing outputs have correct shapes, top-k semantics, and capacity accounting.
- matched-compute counters are stable across comparable router variants.
- routing metrics compute expected values on synthetic fixtures.

- **Integration tests**
- one full CPU train, eval, checkpoint, and resume cycle on a tiny corpus.
- config composition works for all main experiment presets.
- MLflow logging records params, metrics, and artifacts with the expected names.
- tokenizer training, loading, and dataset packing are reproducible.
- Python launcher files correctly validate arguments and dispatch the intended train, eval, and sweep workflows.

- **Smoke and distributed tests**
- dense baseline smoke run.
- top-2 MoE smoke run.
- reflected-router smoke run.
- 2-rank Linux DDP smoke run with `torchrun`.
- downstream eval adapter smoke run on one checkpoint.
- dataloader throughput smoke check confirms that tokenization, caching, packing, and batching are not the first-order bottleneck.

## Assumptions and Defaults

- Primary target is **Linux GPU cloud execution**; Windows is for local development and CPU validation only.
- Default Python is **3.11 via uv-managed interpreter**, not the current system Python 3.10.
- Default tracker is **MLflow**, with optional TensorBoard mirroring.
- Default code layout is **`src/` layout**, consistent with pytest and Python packaging guidance.
- Default baseline corpus stack is **TinyStories or another tiny local corpus for smoke**, **`HuggingFaceFW/fineweb` with `sample-10BT` for main runs**, and **`sample-100BT` only as an optional stronger-validation stage**.
- Full FineWeb is considered unnecessarily large and expensive for the default implementation path and is therefore out of scope unless the research plan is later revised.
- Default orchestration uses **Python launcher files**, not shell scripts.
- Default main experiments should favor **cached tokenized shards and packed sequences** over repeated raw-text preprocessing for better throughput and cleaner matched-compute comparisons.
- The repo will be **package-structured from day one**, not notebook-first.
- The method section must later define `routing churn` mathematically and specify the exact non-inferiority threshold before decisive experiments.
