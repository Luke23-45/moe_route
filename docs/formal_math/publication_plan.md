# Publication Plan for the Reflected MoE Formal Math

This plan describes how to turn the formal reflected architecture into
open-source documentation and a publication-ready method/theory section.

## 1. Method Section Structure

The main paper should use the following structure.

1. **Decoder-only MoE setting**
   - Define the Transformer block.
   - Define which feed-forward layers become MoE layers.
   - State that the attention path is unchanged.

2. **Reflected score**
   - Define gate logits `a_{l,t,e}`.
   - Define expert bias `b_{l,e}`.
   - Define nonnegative pressure `q_{l,e}`.
   - Define penalty `phi_{l,e}(q_{l,e})`.
   - Define `s_{l,t,e} = a_{l,t,e} + b_{l,e} - rho_l phi_{l,e}(q_{l,e})`.

3. **Routing modes**
   - Mode A: dense reflected routing.
   - Mode B: sparse reflected routing.
   - Explain that sparse mode is an implementation/deployment mode, not a
     separate reflected architecture.

4. **Pressure update**
   - Define routed mass.
   - Define target expert capacity fraction.
   - Define projected update.
   - State that pressure update is outside autograd in the implementation.

5. **Training objective**
   - Main method: language-model loss plus zero auxiliary routing loss.
   - Optional ablation: z-loss on raw gate logits only.
   - Do not describe z-loss as required for reflected routing.

6. **Diagnostics**
   - Load fraction.
   - Raw load fraction.
   - overflow.
   - drop rate.
   - matched compute fraction.
   - pressure trajectory.

## 2. Theory Appendix Structure

The theory appendix should separate exact implemented facts from idealized
population analysis.

1. **Exact finite-batch identities**
   - Softmax and normalized sigmoid rows sum to one.
   - Dense routed mass sums to the number of tokens.
   - Sparse raw demand divided by `K` sums to the number of tokens.
   - Projection keeps pressure nonnegative.

2. **Entropy-regularized routing identity**
   - For fixed pressure and softmax gate, dense routing is the unique maximizer
     of a token-wise entropy-regularized linear objective.

3. **Dual-control interpretation**
   - Pressure is a scaled nonnegative dual-control state for expert capacity
     violations.
   - Complementarity is an equilibrium condition, not something guaranteed at
     every minibatch step.

4. **Population reflected dynamics**
   - Freeze the gate and expert parameters.
   - Replace minibatch mass by expected mass under a stationary token
     distribution.
   - Study the reflected projected dynamical system on `R_+^E`.

5. **Stability theorem**
   - State assumptions explicitly.
   - Prove nonnegativity, boundedness under subcritical load, and convergence
     to the KKT/minimizer set in the convex-potential softmax case.

6. **Limitations**
   - No theorem for full nonconvex LM training.
   - Sparse top-k routing introduces discontinuities.
   - Capacity admission uses priority sorting, so sparse mode needs separate
     finite-batch accounting rather than the same differentiable proof as dense
     softmax.

## 3. Proofreading Checklist

Before using this in a paper, check every claim against the current code.

1. `kind: reflected_v2` is the only reflected router kind.
2. Dense mode dispatches every token to every expert.
3. Sparse mode uses top-k after pressure-adjusted scoring.
4. Softmax sparse mode takes top-k of reflected scores.
5. Sigmoid sparse mode takes top-k of sigmoid-reflected scores.
6. Capacity in sparse mode is `max(int(capacity_factor * T * K / E), 1)`.
7. Sparse pressure update uses `raw_load / K`, not accepted load.
8. Dense pressure update uses soft routed mass.
9. Pressure update is only performed when the router module is in training mode.
10. z-loss, when enabled, is computed from raw gate logits, not pressured scores.

## 4. Suggested Open-Source Artifacts

1. Keep `reflected_moe_formalism.md` as the canonical math spec.
2. Add an implementation cross-reference table to the README once the code
   stabilizes.
3. Add a diagram showing the flow:
   `hidden state -> gate logits -> reflected score -> dense/sparse routing -> expert output -> pressure update`.
4. Add tests that assert the formulas in the formalism:
   - pressure projection,
   - mass conservation in dense mode,
   - raw sparse demand scaling,
   - raw-logit z-loss.

## 5. Claims That Should Not Be Made

Do not claim:

1. Reflected routing proves end-to-end language-model convergence.
2. Reflected sparse routing is fully differentiable through top-k selection.
3. Dense reflected routing is compute-matched to sparse top-2 without explicit
   accounting.
4. Pressure guarantees perfect balance at every minibatch.
5. The method universally dominates Switch, ST-MoE, Expert Choice, Soft MoE, or
   DeepSeek-style routing.

The defensible claim is narrower and stronger:

> Reflected routing gives an explicit pressure-controlled mechanism for MoE load
> regulation, with formal router-level stability under controlled assumptions
> and empirical evaluation under matched-compute conditions.
