Yes. I went back through the primary MoE papers, and the clean conclusion is:

**Reflected should be defined as a native reflected controller, not as a `TopKRouter` variant.**
That is consistent with the current MoE design space: Switch and ST-MoE use token-choice routing plus auxiliary load balancing and router z-loss; Expert Choice flips the routing direction and gives each expert a fixed bucket size; Soft MoE replaces hard sparse routing with fully differentiable soft assignment; and auxiliary-loss-free / loss-free balancing updates expert-wise biases from recent load instead of relying on interference-prone auxiliary gradients. A recent 2025 router paper also warns that balancing too aggressively toward uniformity can create inconsistent routing and redundant knowledge, so the goal should be controlled, bounded load balancing rather than entropy flattening. 

Below is the **paper-ready formal definition** I would use.

---

# Reflected Controller MoE

## 1. Model

Let a decoder-only Transformer have (L) layers.
At layer (\ell), let the hidden state for token (t) be
[
h_t^{(\ell)} \in \mathbb R^d.
]

Let that layer contain (E_\ell) experts
[
f_{\ell,1},\dots,f_{\ell,E_\ell}.
]

For each layer (\ell), define:

* a gate network
  [
  G_\ell:\mathbb R^d \to \mathbb R^{E_\ell},
  ]
* an expert-bias vector
  [
  b_\ell \in \mathbb R^{E_\ell},
  ]
* a nonnegative pressure state
  [
  q_\ell \in \mathbb R_+^{E_\ell},
  ]
* a target capacity vector
  [
  c_\ell \in \mathbb R_+^{E_\ell},
  \qquad \sum_{e=1}^{E_\ell} c_{\ell,e} = K_\ell T
  ]
  if the layer is designed to activate, on average, (K_\ell) experts per token over a batch of (T) tokens.

The reflected router is defined layerwise. For each token (t) and expert (e), define the raw affinity
[
a_{\ell,t,e} := \big(G_\ell(h_t^{(\ell)})\big)_e.
]

The **reflected score** is
[
s_{\ell,t,e}
============

a_{\ell,t,e} + b_{\ell,e} - \rho_\ell,\phi_{\ell,e}(q_{\ell,e}),
]
where (\rho_\ell>0) is a router scale and (\phi_{\ell,e}) is a monotone load penalty.
The clean default choice is
[
\phi_{\ell,e}(q_{\ell,e})=\frac{q_{\ell,e}}{\mu_{\ell,e}^\beta+\varepsilon},
]
with (\mu_{\ell,e}>0), (\beta>0), and small (\varepsilon>0).

The training-time routing distribution is
[
p_{\ell,t,e}
============

\frac{\exp(s_{\ell,t,e}/\tau_\ell)}
{\sum_{j=1}^{E_\ell}\exp(s_{\ell,t,j}/\tau_\ell)},
]
with temperature (\tau_\ell>0).

The MoE layer output is
[
y_t^{(\ell)}=\sum_{e=1}^{E_\ell}\tilde p_{\ell,t,e},f_{\ell,e}(h_t^{(\ell)}).
]

Here (\tilde p_{\ell,t,e}) is either:

1. the full soft routing probability (p_{\ell,t,e}) in the dense training variant, or
2. a sparsified and renormalized version for efficient deployment.

The **core architecture** is the reflected controller; sparsification is only an implementation choice.

---

## 2. Reflected pressure dynamics

Define the routed mass assigned to expert (e) in a batch:
[
m_{\ell,e} := \sum_{t=1}^T \tilde p_{\ell,t,e}.
]

The pressure state is updated by **projected reflected ascent**:
[
q_{\ell,e}^{(n+1)}
==================

\Pi_{[0,\infty)}
!\left(
q_{\ell,e}^{(n)}
+
\eta_\ell\big(m_{\ell,e}^{(n)}-c_{\ell,e}\big)
\right),
]
where (\eta_\ell>0) is the pressure step size and
[
\Pi_{[0,\infty)}(z)=\max{0,z}.
]

This is the key reflected mechanism: underloaded experts keep zero pressure, while overloaded experts accumulate pressure and become less attractive on future batches.

---

## 3. Primal-dual interpretation

The router can be written as an entropy-regularized constrained assignment problem.

For a batch (X=(x_1,\dots,x_T)), define the routing matrix
[
\Pi_\ell = (\Pi_{\ell,t,e}) \in \mathbb R_+^{T\times E_\ell}.
]

Let
[
\mathcal C_\ell
===============

\Big{
\Pi_\ell:;
\sum_{e=1}^{E_\ell}\Pi_{\ell,t,e}=1 \ \forall t,;
\sum_{t=1}^T\Pi_{\ell,t,e}\le c_{\ell,e}\ \forall e
\Big}.
]

Then the reflected router solves, approximately or exactly depending on implementation,
[
\Pi_\ell^\star
==============

\arg\max_{\Pi_\ell\in\mathcal C_\ell}
\left[
\sum_{t=1}^T\sum_{e=1}^{E_\ell}\Pi_{\ell,t,e}\big(a_{\ell,t,e}+b_{\ell,e}\big)
+
\tau_\ell \sum_{t=1}^T H(\Pi_{\ell,t,:})
\right],
]
where
[
H(\Pi_{\ell,t,:})=-\sum_{e=1}^{E_\ell}\Pi_{\ell,t,e}\log \Pi_{\ell,t,e}
]
is token-wise entropy.

The pressure vector (q_\ell) is the reflected dual state associated with the capacity constraints.
At equilibrium, the complementary-slackness conditions are
[
q_{\ell,e}\ge 0,\qquad
m_{\ell,e}\le c_{\ell,e},\qquad
q_{\ell,e}\big(m_{\ell,e}-c_{\ell,e}\big)=0.
]

That is the precise mathematical meaning of “reflected” in this MoE architecture.

---

## 4. Training objective

The **core paper version** should use
[
\mathcal L_{\text{core}} = \mathcal L_{\text{task}},
]
with no auxiliary balance loss and no router z-loss in the main method.

If a stability ablation is needed, add z-loss only on the **raw gate logits** (a_{\ell,t,:}), not on the pressured logits (s_{\ell,t,:}):
[
\mathcal L_{z,\ell}
===================

\frac{1}{T}
\sum_{t=1}^T
\left(
\log\sum_{e=1}^{E_\ell}\exp(a_{\ell,t,e})
\right)^2.
]

That matches the role z-loss is used for in ST-MoE: it regularizes the router logits themselves rather than the pressure-controlled scores. 

If you want a tiny diagnostic balance term, keep it sequence-local and very small:
[
\mathcal L_{\text{seq-bal},\ell}
================================

\sum_{e=1}^{E_\ell}
\left(
\frac{m_{\ell,e}}{T} - \frac{c_{\ell,e}}{T}
\right)^2.
]
But this should be an ablation, not the core claim.

The main method should therefore be **auxiliary-loss-free by default**, which is aligned with recent load-balancing practice that replaces large auxiliary losses with dynamic expert-wise bias control. ([arXiv][1])

---

## 5. Inference rule

For inference, use the same reflected score
[
s_{\ell,t,e}
============

a_{\ell,t,e}+b_{\ell,e}-\rho_\ell\phi_{\ell,e}(q_{\ell,e}).
]

Then choose one of two deployment modes:

**Mode A: dense reflected routing**
Use the full softmax probabilities (p_{\ell,t,e}).

**Mode B: sparse reflected deployment**
Keep only the largest (K_\ell) entries of (p_{\ell,t,:}), renormalize, and dispatch only those experts.

The first mode is the cleanest mathematical definition. The second is the compute-efficient implementation.

---

## 6. What makes this architecture different from prior MoE families

This matters for the paper’s framing.

It is **not Switch**: Switch uses token-choice top-1 routing plus auxiliary balance loss, and it explicitly adds a load-balancing objective to push routing toward uniformity. 

It is **not ST-MoE**: ST-MoE still keeps the router logits under an explicit z-loss and total loss of the form task loss plus auxiliary load balance plus z-loss. 

It is **not Expert Choice**: Expert Choice flips the direction so experts choose tokens, with fixed bucket size per expert. ([arXiv][2])

It is **not Soft MoE**: Soft MoE uses soft assignment over all tokens to every expert combination and is fully differentiable, specifically to address instability, token dropping, and scaling issues. ([arXiv][3])

It is **closest in spirit to loss-free balancing**, but with an explicit reflected pressure state rather than only a heuristic bias update. Loss-Free Balancing and DeepSeek-V3 update per-expert biases from recent load and report better load balance without interference gradients; DeepSeek-V3 still adds a very small sequence-wise balance loss only to prevent extreme within-sequence imbalance. ([arXiv][1])

---

## 7. What the theory claim should be

The safe theorem statement for the paper is:

**Theorem (stationary reflected equilibrium, controlled setting).**
Under fixed token-affinity statistics, subcritical load, and mild regularity assumptions on the gate outputs, the reflected pressure dynamics admit a stable equilibrium characterized by the KKT conditions of the capacity-constrained entropy-regularized routing problem.

That theorem is the right scope. It proves the controller behavior, not end-to-end LM training convergence.

---

## 8. What I would keep fixed in v1

For the first publishable version, I would keep:

1. **token-blind pressure** in the main method,
2. **zero auxiliary loss** in the core method,
3. **z-loss only as an ablation on raw gate logits**, if used at all,
4. **sparse deployment as an implementation choice**, not the conceptual core,
5. **capacity-control through the reflected state**, not through a standard load-balancing auxiliary objective.

That is the cleanest architecture and the easiest to defend in review.

---

## 9. Final formal definition in one line

If you need a one-sentence definition for the manuscript, use this:

> **ReflectedMoE is an entropy-regularized, capacity-constrained MoE router whose expert-selection scores combine token affinity with a reflected nonnegative expert-pressure state, where the pressure is updated by projected dual ascent from recent routed mass and the core model is trained without auxiliary routing losses.**

That is the complete architecture I would use for the paper.

I can turn this next into a manuscript-ready **Method section with Definition / Proposition / Algorithm blocks** and make it look like a proper paper draft.

[1]: https://arxiv.org/abs/2408.15664 "[2408.15664] Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts"
[2]: https://arxiv.org/abs/2202.09368 "[2202.09368] Mixture-of-Experts with Expert Choice Routing"
[3]: https://arxiv.org/html/2308.00951v2 "From Sparse to Soft Mixtures of Experts"
