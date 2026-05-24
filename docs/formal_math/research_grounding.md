# Research Grounding for Reflected MoE Routing

This note records the literature context used to write the formal reflected
router specification. It is intentionally conservative: it only states claims
needed to position the current implementation.

## Primary References

1. Shazeer et al., **Outrageously Large Neural Networks: The Sparsely-Gated
   Mixture-of-Experts Layer**, 2017.
   URL: https://arxiv.org/abs/1701.06538
2. Lepikhin et al., **GShard: Scaling Giant Models with Conditional Computation
   and Automatic Sharding**, 2020.
   URL: https://arxiv.org/abs/2006.16668
3. Fedus et al., **Switch Transformers: Scaling to Trillion Parameter Models
   with Simple and Efficient Sparsity**, 2021.
   URL: https://arxiv.org/abs/2101.03961
4. Zoph et al., **ST-MoE: Designing Stable and Transferable Sparse Expert
   Models**, 2022.
   URL: https://arxiv.org/abs/2202.08906
5. Zhou et al., **Mixture-of-Experts with Expert Choice Routing**, 2022.
   URL: https://arxiv.org/abs/2202.09368
6. Puigcerver et al., **From Sparse to Soft Mixtures of Experts**, 2023.
   URL: https://arxiv.org/abs/2308.00951
7. Wang et al., **Auxiliary-Loss-Free Load Balancing Strategy for
   Mixture-of-Experts**, 2024.
   URL: https://arxiv.org/abs/2408.15664
8. DeepSeek-AI, **DeepSeek-V3 Technical Report**, 2024.
   URL: https://arxiv.org/abs/2412.19437

## What Prior Work Establishes

Sparse MoE layers increase parameter count without activating every expert for
every token. In the classic sparse-gated formulation, a trainable gate selects a
sparse expert set for each input token. GShard and Switch Transformer make this
idea practical at Transformer scale using token-choice routing, expert capacity,
and load-balancing losses.

ST-MoE keeps sparse expert routing but emphasizes stability and transfer. It is
relevant because it treats router regularization, including z-loss, as an
important part of stable sparse expert training.

Expert Choice Routing reverses the assignment direction: experts choose tokens
under fixed expert capacities. This avoids some token-choice imbalance but is a
different routing family from the reflected controller in this repo.

Soft MoE removes hard token-to-expert dispatch and uses soft token mixtures,
which avoids token dropping and gives a differentiable assignment mechanism.
The reflected dense mode is closer to this soft-routing spirit than to Switch,
but the reflected pressure state is a different load-control mechanism.

Auxiliary-loss-free load balancing and DeepSeek-V3 are relevant because they
move load control away from a large auxiliary balancing gradient. They use
expert-wise bias adjustment or routing bias strategies so the main gate can
focus more on affinity. The reflected router follows the same broad direction,
but uses an explicit nonnegative pressure state with projected updates.

## What Reflected MoE Adds

The reflected router in this repository is defined by three coupled parts:

1. A token-expert affinity score from a learned gate.
2. A nonnegative expert pressure state.
3. A reflected pressure update that raises pressure on overloaded experts and
   clamps underloaded pressure at zero.

The reflected score is

```math
s_{t,e} = a_{t,e} + b_e - \rho \phi_e(q_e).
```

The pressure penalty is monotone in `q_e`, so an expert that repeatedly receives
excess routed mass becomes less attractive in later batches.

## What Reflected MoE Is Not

It is not a legacy `TopKRouter` variant. In the current code, reflected routing
is implemented by `ReflectedController`, selected through `kind: reflected_v2`.

It is not Switch routing. Switch is sparse top-1 token-choice routing with
capacity handling and auxiliary balancing. Reflected dense routing activates all
experts with soft weights; reflected sparse routing uses top-k after applying a
pressure-adjusted score.

It is not Expert Choice routing. Experts do not choose their token buckets.
Tokens still choose experts, but their scores are load-aware through pressure.

It is not a proof of full Transformer training convergence. The proof package
can establish router-level invariants, fixed-score optimization identities, and
population pressure-dynamics stability under simplifying assumptions. It cannot
honestly prove nonconvex end-to-end language-model training convergence.

## Publication Positioning

The safest publication claim is:

> Reflected MoE is a load-aware MoE routing architecture whose expert-selection
> scores combine token affinity with a nonnegative reflected pressure state. The
> pressure state is updated by projected ascent from recent routed mass, giving
> a router-level mechanism for capacity control without relying on a large
> auxiliary load-balancing loss.

The main empirical claim should remain matched-compute and controlled:

> Under matched architecture and matched compute, reflected routing should
> improve routing stability and load behavior relative to standard token-choice
> top-k routing while preserving language-model quality.
