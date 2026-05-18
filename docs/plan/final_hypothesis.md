# Final Hypothesis

## Status

This document is the finalized working hypothesis for the project. It is intended to be the foundation for the method design, experiment plan, ablation plan, and thesis or paper framing. Any later implementation or evaluation decision should remain consistent with the definitions and boundaries set here unless this document is explicitly revised.

## Core research claim

The project studies whether a load-aware reflected routing mechanism can improve the behavior of Mixture-of-Experts language models without sacrificing model quality.

The central claim is:

> In a decoder-only Transformer that uses MoE only in the feed-forward sublayers, replacing standard token-choice top-k routing with a load-aware reflected router that combines token-expert affinity with a dynamic expert-pressure state will produce more stable and balanced expert utilization under matched architecture and matched compute, while preserving or improving language-model quality.

This claim is deliberately narrow. It does not claim universal superiority across all MoE systems. It claims a specific advantage in a controlled comparison against standard top-k routing in the target setting of this project.

## Final primary hypothesis

### H1. Main hypothesis

Under a controlled comparison in which the backbone architecture, parameter scale, expert count, activated FLOPs, training data, optimizer family, and training budget are held fixed, a load-aware reflected MoE router will outperform standard token-choice top-k routing on routing-behavior metrics and will be non-inferior on language-model quality metrics.

More specifically, the proposed router is expected to:

1. reduce expert-load imbalance,
2. reduce routing instability across training,
3. reduce expert overload and token-drop behavior,
4. reduce dependence on a large auxiliary load-balancing loss,
5. preserve or improve validation perplexity and downstream task performance.

### Why this is the correct level of claim

This hypothesis is strong enough to support a serious research contribution, but restricted enough to remain defensible:

1. it targets MoE routing rather than all sparse modeling,
2. it compares against the most relevant standard baseline,
3. it requires matched-compute evaluation rather than unconstrained comparison,
4. it allows the method to fail on quality if routing gains do not translate into useful model behavior,
5. it avoids claiming frontier-scale success before evidence exists.

## Formal method-aligned hypothesis

Let the proposed routing score for expert \(e\) be

\[
s_e(x, q) = a_e(x) + b_e - \phi_e(q),
\]

where:

1. \(a_e(x)\) is the token-expert affinity term produced by the gating network,
2. \(b_e\) is an expert-specific static bias or prior term,
3. \(q_e\) is the dynamic pressure state associated with expert \(e\); at the hypothesis stage it is an abstract load-aware state variable, while its exact implementation may later be instantiated as a virtual backlog, normalized pressure variable, exponential-moving-average load state, or another equivalent reflected update quantity,
4. \(\phi_e(q)\) is a monotone load penalty that increases when the expert becomes relatively overloaded.

The control condition is a standard token-choice top-k router using the same Transformer backbone, the same number of experts, the same dataset, the same optimization setup, and the same training budget.

The formal empirical hypothesis is:

\[
H_1:
\quad
\text{The proposed load-aware reflected router improves expert-balance and routing-stability metrics}
\]
\[
\text{relative to standard top-k routing, without causing unacceptable degradation in model quality}
\]
\[
\text{under a matched-compute comparison.}
\]

The formal theoretical hypothesis is:

\[
H_{1,\mathrm{theory}}:
\quad
\text{Under the simplified stationary population model and subcritical load, the reflected pressure dynamics}
\]
\[
\text{can be analyzed through a convex-potential formulation and are expected to admit a stable equilibrium}
\]
\[
\text{characterized by the minimizer set of that potential.}
\]

The theoretical statement is not the full project claim. It is a supporting claim about the router dynamics under simplifying assumptions, not a proof of end-to-end LLM training convergence.

## Null hypotheses

### H0. Primary null hypothesis

\[
H_0:
\quad
\text{Under matched architecture and matched compute, the load-aware reflected router does not improve}
\]
\[
\text{expert-balance or routing-stability metrics relative to standard top-k routing, or any such gains}
\]
\[
\text{are offset by degraded language-model quality or impractical computational overhead.}
\]

### Secondary null hypotheses

1. Any observed balancing improvement is explained mainly by auxiliary regularization rather than by the pressure-based routing mechanism itself.
2. The reflected pressure dynamics do not materially reduce collapse, overload, token drop, or routing churn.
3. The proposed router does not provide a useful convergence-speed advantage in training steps or wall-clock time.
4. Any theoretical stability result under the simplified population model does not translate into meaningful empirical benefit during minibatch training.

## Secondary hypotheses

### H2. Collapse reduction and smoother specialization

Because the pressure state penalizes repeatedly overloaded experts, the router will reduce collapse into a small subset of experts and will encourage more persistent expert specialization patterns over training.

### H3. Earlier gains in routing behavior than in downstream quality

The first measurable improvements should appear in router-side quantities such as load variance, Gini coefficient, token-drop rate, overload incidence, and routing churn. Improvements in perplexity or downstream benchmarks may appear later and may be smaller in magnitude.

### H4. Reduced need for strong balancing losses

The proposed router should remain balanced with a smaller auxiliary load-balancing coefficient than standard top-k routing requires for similar expert-usage behavior.

### H5. Practical boundedness of pressure dynamics

Although stochastic minibatch training will not exactly follow the deterministic reflected flow, the learned pressure dynamics should remain practically bounded and should reduce persistent expert overload events.

## Scope boundaries

### What this project does claim

1. the method is a new load-aware routing formulation for MoE feed-forward layers,
2. the method has a clear dynamical interpretation through reflected expert-pressure updates,
3. the method should improve routing behavior under controlled matched-compute comparison,
4. quality may be preserved or improved as a result of better routing behavior,
5. the method is worthy of evaluation against standard MoE routing baselines.

### What this project does not claim

1. that the method solves MoE routing in all regimes,
2. that it outperforms every routing method in the literature,
3. that it replaces attention or redesigns the entire Transformer,
4. that a simplified stability theorem proves full language-model training convergence,
5. that results at small or medium scale automatically imply frontier-scale gains.

These boundaries are important. If the project later overstates its claim beyond this scope, the argument becomes weaker rather than stronger.

## Assumptions that must remain explicit

The main hypothesis relies on the following assumptions:

1. the MoE mechanism is introduced only in feed-forward sublayers in the initial study,
2. the comparison is made under a fixed or carefully normalized compute budget,
3. routing overhead is measured rather than ignored,
4. expert capacities are comparable across methods,
5. data preprocessing and tokenization are identical across compared systems,
6. evaluation seeds and training randomness are controlled tightly enough to make the comparison interpretable,
7. the pressure update mechanism is implemented faithfully enough to represent the proposed method rather than an unrelated heuristic.

If any of these assumptions are violated, the empirical interpretation of the hypothesis becomes weaker.

## Operational definitions

To prevent ambiguity later, the main terms in the hypothesis are defined here.

### Matched architecture

Matched architecture means the compared systems share:

1. the same decoder-only backbone design,
2. the same depth and width,
3. the same MoE insertion pattern,
4. the same number of experts,
5. the same parameter budget up to unavoidable implementation-level differences.

### Matched compute

Matched compute means the comparison holds constant, or normalizes carefully for:

1. activated FLOPs per token,
2. total training tokens,
3. optimization schedule,
4. total training steps or total optimization budget,
5. expert capacity setting,
6. inference routing sparsity for the main comparison.

If exact equality is impossible, any residual mismatch must be reported and justified.

### Stable expert utilization

Stable expert utilization means expert assignments do not fluctuate erratically across neighboring training steps and do not repeatedly collapse into a narrow subset of experts without recovery.

### Balanced expert utilization

Balanced expert utilization means the distribution of routed mass or routed tokens across experts is more even, as measured by quantities such as Gini coefficient, variance, or coefficient of variation, without trivializing specialization.

### Non-inferior quality

Non-inferior quality means the proposed method does not degrade validation perplexity or downstream task performance beyond a pre-specified tolerance margin determined before the decisive experiment is run.

The margin must be chosen in advance and justified in the experiment plan. It must not be chosen after observing results.

## Variables

### Independent variable

The independent variable is the routing mechanism:

1. standard token-choice top-k routing,
2. proposed load-aware reflected routing.

### Primary dependent variables

The primary dependent variables are divided into three groups.

Quality metrics:

1. validation perplexity,
2. downstream language-task accuracy or score on a fixed evaluation suite.

Routing metrics:

1. expert-load variance,
2. expert-load coefficient of variation,
3. expert-load Gini coefficient,
4. token-drop rate,
5. overload incidence,
6. routing entropy,
7. routing churn across steps, checkpoints, or epochs.

For rigor, routing churn must later be given an explicit mathematical definition in the methods section before the decisive experiment is run.

Efficiency metrics:

1. convergence speed in steps,
2. convergence speed in wall-clock time,
3. throughput in tokens per second,
4. routing overhead relative to the baseline.

### Controlled variables

The following variables must be controlled or reported explicitly:

1. backbone architecture,
2. total parameter count,
3. number of experts,
4. routing sparsity level at comparison time,
5. activated FLOPs,
6. dataset, data filtering, and tokenization,
7. optimizer, learning-rate schedule, and regularization,
8. batch size and sequence length,
9. expert capacity factor,
10. total training tokens,
11. total training steps,
12. random seed policy and number of runs.

## Acceptance criteria

The primary hypothesis should be treated as supported only if all of the following are satisfied in the main matched-compute comparison:

1. the proposed router improves at least two routing-behavior metrics,
2. at least one of those improvements is a direct load-balance metric such as Gini coefficient, variance, or coefficient of variation,
3. token drop, overload, or collapse behavior is reduced relative to standard top-k routing,
4. validation perplexity is non-inferior or better,
5. the computational overhead remains small enough that the method is still practically competitive.

The primary hypothesis should be treated as strongly supported if downstream task performance also improves materially.

The primary hypothesis should be treated as only partially supported if routing improves clearly but quality remains ambiguous.

The primary hypothesis should be treated as not supported if any of the following occurs:

1. balance improves but validation perplexity worsens beyond the pre-registered non-inferiority margin,
2. quality is unchanged but routing overhead is large enough to remove practical value,
3. the method balances experts only by damaging useful specialization,
4. the improvement appears only under a narrowly tuned auxiliary loss that the baseline did not equally receive.

## Failure conditions and interpretation rules

The project should not force a positive conclusion if the evidence points elsewhere.

The following cases must be interpreted honestly:

1. If balance improves but quality does not, the result supports the router as a systems or stability contribution, not necessarily as a modeling-quality contribution.
2. If quality improves but balance does not, the mechanism behind the gain must be re-examined before attributing the result to reflected load awareness.
3. If the method helps only in a narrow hyperparameter region, robustness becomes a central weakness rather than a minor limitation.
4. If the method requires higher routing overhead than expected, practical deployment value may be limited even if the hypothesis is partly supported.

## Why this version is final-grade

This version is stronger than the earlier draft because it:

1. is self-contained,
2. aligns with the mathematical formulation already developed for the router,
3. separates the empirical claim from the theoretical supporting claim,
4. defines the comparison conditions more tightly,
5. defines what counts as success, partial success, or failure,
6. reduces the chance of later goalpost shifting,
7. remains ambitious without becoming careless.

## Official working hypothesis

This should be treated as the project's official working hypothesis:

> A load-aware reflected MoE router that augments token-expert affinity with a dynamic expert-pressure state will improve expert utilization stability and balance, reduce overload-related routing failures, and preserve or improve language-model quality under matched architecture and matched compute relative to standard token-choice top-k routing.

## Short thesis-ready version

If a shorter statement is needed for a proposal or thesis synopsis, use:

> Under matched architecture and matched compute, load-aware reflected routing for MoE feed-forward layers yields more stable and balanced expert utilization than standard top-k routing while preserving or improving language-model quality.
