# Formal Math Package for Reflected MoE

This directory contains the formal mathematical specification for the reflected
router currently implemented in the repository.

The documents are written to support open-source documentation and a future
paper or thesis method section. They distinguish three levels of claim:

1. **Implemented architecture**: definitions that match the current code in
   `src/moe_route/routing/reflected_controller.py` and
   `src/moe_route/models/moe.py`.
2. **Router-level theory**: properties of the reflected score, routing map,
   pressure update, and simplified population dynamics.
3. **End-to-end model framing**: the decoder-only Transformer with MoE feed
   forward layers, stated without claiming full training convergence.

## Files

- [research_grounding.md](research_grounding.md): literature context and what
  the reflected router is, and is not, relative to prior MoE routing methods.
- [reflected_moe_formalism.md](reflected_moe_formalism.md): notation,
  definitions, algorithm blocks, propositions, and proofs for the current
  architecture.
- [publication_plan.md](publication_plan.md): a detailed plan for turning the
  formalism into an open-source method section and publication-ready theory
  appendix.

## Code Alignment

The current reflected implementation has one router kind, `reflected_v2`, with
two modes:

- `routing_mode: dense`, configured by `configs/router/reflected.yaml`.
- `routing_mode: sparse`, configured by `configs/router/reflected_sparse.yaml`.

Both modes share the same reflected score

```math
s_{t,e} = a_{t,e} + b_e - \gamma q_e,
```

where `a` is the raw gate logit, `b` is an expert bias, and `q` is the
nonnegative routed-expert pressure state.

The reflected revision in `docs/implemention/x1.md` is now incorporated into
the formalism and implementation:

- pressure feedback uses continuous routed probability mass,
- sparse reflected updates are drop-aware after capacity admission,
- reflected layers can include always-on shared experts,
- the reflected penalty is parameterized by an absorbed multiplier `gamma`,
  with pressure updates scaled by `1 / sqrt(d_model)` and a phase-aware decay
  after a warmup horizon.

These changes apply only to the reflected router path. Standard dense, top-1,
and top-2 baselines remain unchanged.

## Scope Boundary

The formal results here are router-level results. They support the architecture
and its pressure-control interpretation, but they do not prove global
convergence of decoder-only language-model training.
