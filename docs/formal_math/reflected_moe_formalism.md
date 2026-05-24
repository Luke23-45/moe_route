# Reflected MoE Formalism

This document defines the reflected architecture currently implemented in this
repository after applying the `docs/implemention/x1.md` revisions. The changes
in this document apply only to the proposed reflected routers. Standard dense,
top-1, and top-2 baselines are unchanged.

## 1. Notation

Let a decoder-only Transformer process a batch of `B` sequences of length `S`.
Inside an MoE layer, flatten the token dimension:

```math
T = B S.
```

For reflected MoE layer `l`, let

```math
h_{l,t} \in \mathbb{R}^d, \qquad t = 1,\dots,T
```

be the hidden state entering the feed-forward sublayer.

The layer contains:

- `E_l` routed experts

```math
f^{route}_{l,1}, \dots, f^{route}_{l,E_l}: \mathbb{R}^d \to \mathbb{R}^d
```

- `K^s_l` shared experts

```math
f^{shared}_{l,1}, \dots, f^{shared}_{l,K^s_l}: \mathbb{R}^d \to \mathbb{R}^d
```

The reflected router has layer-local parameters and state:

```math
G_l: \mathbb{R}^d \to \mathbb{R}^{E_l}, \qquad
b_l \in \mathbb{R}^{E_l}, \qquad
q_l \in \mathbb{R}_+^{E_l}.
```

The routed-expert target remains uniform:

```math
\mu_{l,e} = 1 / E_l.
```

The reflected-only hyperparameters are:

```math
\tau_l > 0 \qquad \text{(routing temperature)},
```

```math
\gamma_l > 0 \qquad \text{(absorbed pressure multiplier)},
```

```math
\eta^{base}_l > 0 \qquad \text{(base pressure step size)},
```

```math
\kappa_l \ge 1 \qquad \text{(phase-schedule warmup horizon)},
```

```math
\delta_l \ge 0 \qquad \text{(pressure decay)}.
```

The implementation uses the width-scaled base step size

```math
\bar \eta_l = \eta^{base}_l / \sqrt{d},
```

and the phase-aware effective step size

```math
\eta_l(t) = \bar \eta_l \cdot \min\left(1, \sqrt{\kappa_l / \max(t,1)}\right).
```

Thus the reflected update starts at the width-scaled base rate and decays after
the warmup horizon `\kappa_l`.

This is the reflected-only reduction suggested in `x1.md`: the earlier
`(\rho, \beta, \epsilon)` decomposition is absorbed into one multiplier
`\gamma_l`, and the remaining reflected gain follows a width-aware and
phase-aware schedule.

## 2. Reflected Score

For token `t` and routed expert `e`, define the raw affinity

```math
a_{l,t,e} = (G_l h_{l,t})_e.
```

The reflected score is

```math
s_{l,t,e} = a_{l,t,e} + b_{l,e} - \gamma_l q_{l,e}.
```

Thus a routed expert with larger pressure receives a uniformly reduced score
for every token in that layer.

## 3. Full Continuous Routing Probabilities

The router always forms a continuous full-distribution signal over routed
experts before any sparse truncation.

### 3.1 Softmax Gate

```math
p_{l,t,e}
  = \frac{\exp(s_{l,t,e} / \tau_l)}
         {\sum_{j=1}^{E_l} \exp(s_{l,t,j} / \tau_l)}.
```

### 3.2 Normalized Sigmoid Gate

Define

```math
u_{l,t,e} = \sigma(s_{l,t,e} / \tau_l),
```

then normalize row-wise:

```math
p_{l,t,e}
  = \frac{u_{l,t,e}}{\sum_{j=1}^{E_l} u_{l,t,j}}.
```

For both gate families,

```math
\sum_{e=1}^{E_l} p_{l,t,e} = 1.
```

This full probability matrix is the continuous feedback signal introduced in
response to the professor's first recommendation.

## 4. Dense Reflected Routing

In dense reflected mode, every token dispatches to every routed expert with

```math
w^{route}_{l,t,e} = p_{l,t,e}.
```

The routed contribution is

```math
y^{route}_{l,t}
  = \sum_{e=1}^{E_l} w^{route}_{l,t,e} f^{route}_{l,e}(h_{l,t}).
```

The shared contribution is

```math
y^{shared}_{l,t}
  = \sum_{j=1}^{K^s_l} f^{shared}_{l,j}(h_{l,t}).
```

The MoE output is

```math
y_{l,t} = y^{shared}_{l,t} + y^{route}_{l,t}.
```

The routed feedback mass is

```math
m_{l,e} = \sum_{t=1}^{T} p_{l,t,e}.
```

Hence

```math
\sum_{e=1}^{E_l} m_{l,e} = T.
```

Shared experts are always on and are not part of the pressure state.

## 5. Sparse Reflected Routing

Sparse reflected mode keeps the same reflected scores and the same full
probability matrix `p`, but executes only `K_l` routed slots per token.

### 5.1 Sparse Selection

For softmax sparse routing,

```math
I_{l,t} = TopK_e(s_{l,t,e}, K_l).
```

For sigmoid sparse routing,

```math
I_{l,t} = TopK_e(u_{l,t,e}, K_l).
```

The execution-time sparse combine weights are:

```math
\bar w_{l,t,e}
  = \frac{\exp(s_{l,t,e}/\tau_l)}
         {\sum_{j \in I_{l,t}} \exp(s_{l,t,j}/\tau_l)},
\qquad e \in I_{l,t}
```

for softmax sparse mode, and

```math
\bar w_{l,t,e}
  = \frac{u_{l,t,e}}
         {\sum_{j \in I_{l,t}} u_{l,t,j}},
\qquad e \in I_{l,t}
```

for sigmoid sparse mode. Unselected weights are zero.

### 5.2 Capacity Admission

Sparse mode uses routed-expert capacity

```math
C_l(T) = \max(\lfloor \alpha_l T K_l / E_l \rfloor, 1),
```

where `\alpha_l` is the capacity factor.

Let `r_{l,t,k}` be the expert chosen in token `t`, slot `k`, and let
`d_{l,t,k} \in \{0,1\}` denote whether that slot is admitted after priority
ordering and capacity truncation.

The execution combine weights are renormalized over accepted slots:

```math
w_{l,t,k}
  = \frac{d_{l,t,k} \bar w_{l,t,k}}
         {\sum_{j=1}^{K_l} d_{l,t,j} \bar w_{l,t,j}}
```

whenever at least one slot is accepted.

The routed sparse contribution is

```math
y^{route}_{l,t}
  = \sum_{k=1}^{K_l} w_{l,t,k} f^{route}_{l,r_{l,t,k}}(h_{l,t}).
```

The total MoE output remains

```math
y_{l,t} = y^{shared}_{l,t} + y^{route}_{l,t}.
```

### 5.3 Continuous, Drop-Aware Feedback Mass

The reflected pressure update no longer uses discrete slot counts.

First gather the pre-top-k full probabilities on the selected experts:

```math
\tilde p_{l,t,k} = p_{l,t,r_{l,t,k}}.
```

Then keep only admitted routed slots and use the final accepted sparse combine
weights:

```math
\hat w_{l,t,k}
  = \frac{d_{l,t,k} \bar w_{l,t,k}}
         {\sum_{j=1}^{K_l} d_{l,t,j} \bar w_{l,t,j}}
```

whenever at least one slot is admitted. If all routed slots are dropped, all
`\hat w_{l,t,k}` are zero.

The pressure feedback mass is

```math
m_{l,e}
  = \sum_{t=1}^{T} \sum_{k=1}^{K_l}
      \hat w_{l,t,k} \mathbf{1}\{r_{l,t,k} = e\}.
```

This is the key reflected update:

1. it is continuous in the full gate probabilities `p`;
2. it is drop-aware because only admitted assignments contribute;
3. it excludes shared experts from the control loop.

The requested continuous mass remains

```math
m^{req}_{l,e} = \sum_{t=1}^{T} p_{l,t,e},
```

which is tracked for diagnostics only.

## 6. Pressure Update

During training, after computing routed feedback mass `m_l`, the reflected
pressure state is updated outside autograd by

```math
\hat m_{l,e} = m_{l,e} / T,
```

```math
q^{+}_{l,e}
  = \max\left(
      0,
      (1 - \delta_l) q_{l,e}
      + \eta_l (\hat m_{l,e} - \mu_{l,e})
    \right).
```

In vector form:

```math
q_l^{+}
  = \Pi_{\mathbb{R}_+^{E_l}}
      \left(
        (1-\delta_l) q_l + \eta_l (\hat m_l - \mu_l)
      \right).
```

The shared experts do not receive pressure coordinates.

## 7. Training Objective

The core reflected method remains auxiliary-loss free:

```math
L = L_{LM}.
```

An optional z-loss ablation can be enabled on raw gate logits `a`:

```math
L_{z,l}
  = \lambda_z \frac{1}{T}
      \sum_{t=1}^{T}
      \left(\log \sum_{e=1}^{E_l} \exp(a_{l,t,e})\right)^2.
```

Then

```math
L = L_{LM} + \sum_l L_{z,l}.
```

The reflected configs default to `z_loss_weight = 0` and
`aux_loss_weight = 0`.

## 8. End-to-End Decoder Block

For a Transformer block with reflected MoE in the feed-forward sublayer:

```math
\tilde h_l = h_l + Attn_l(LN_1(h_l)),
```

```math
h_{l+1}
  = \tilde h_l + ReflectedMoE_l(LN_2(\tilde h_l), q_l).
```

The layer output already includes the sum of shared and routed expert
contributions from Sections 4 and 5.

## 9. Algorithm

### Algorithm 1: Reflected MoE Layer, Training Mode

Inputs: hidden states `h`, routed experts `f^{route}`, shared experts
`f^{shared}`, gate `G`, bias `b`, pressure `q`.

1. Flatten batch and sequence into `T` tokens.
2. Compute raw affinity `a_{t,e} = (G h_t)_e`.
3. Compute reflected score `s_{t,e} = a_{t,e} + b_e - \gamma q_e`.
4. Compute full continuous probabilities `p_t` over routed experts.
5. Execute dense routing or sparse top-k routing.
6. In sparse mode, admit routed slots under capacity and form accepted
   continuous feedback weights `\hat w`.
7. Compute the routed output and add the always-on shared expert output.
8. Form routed feedback mass `m`.
9. If training, update
   `q <- \Pi_{\mathbb{R}_+^E}((1-\delta)q + \eta(m/T - \mu))`.
10. Return output and optional raw-logit z-loss.

## 10. Propositions

### Proposition 1: Pressure Nonnegativity

For any initial `q_l \in \mathbb{R}_+^{E_l}`, every subsequent pressure vector
produced by the update in Section 6 remains in `\mathbb{R}_+^{E_l}`.

**Proof.**
The update applies the coordinatewise projection
`z \mapsto \max(0, z)` to every coordinate. Therefore every output coordinate
is nonnegative. QED.

### Proposition 2: Dense Routed-Mass Conservation

In dense reflected mode,

```math
\sum_{e=1}^{E_l} m_{l,e} = T.
```

**Proof.**
Each token-level probability vector `p_{l,t,:}` is row-normalized, so
`\sum_e p_{l,t,e} = 1`. Summing over tokens gives `T`. QED.

### Proposition 3: Sparse Accepted Feedback Mass Is Bounded by Token Count

In sparse reflected mode,

```math
0 \le \sum_{e=1}^{E_l} m_{l,e} \le T.
```

**Proof.**
For each token, the accepted feedback weights `\hat w_{l,t,:}` either sum to
`1` when at least one routed slot is admitted, or sum to `0` when all routed
slots are dropped. Summing over tokens gives a total in `[0, T]`. QED.

### Proposition 4: Reflected Score Is Monotone in Pressure

For fixed token affinity and bias,

```math
q'_e \ge q_e \implies s_{t,e}(q'_e) \le s_{t,e}(q_e).
```

**Proof.**
The reflected score subtracts `\gamma q_e` with `\gamma > 0`, so increasing
`q_e` can only decrease the score. QED.

### Proposition 5: Shared Experts Are Outside the Pressure Controller

The pressure update in Section 6 depends only on routed-expert feedback mass
`m_l` and has no coordinates for shared experts.

**Proof.**
The controller state is `q_l \in \mathbb{R}_+^{E_l}` and Section 5.3 defines
`m_l` only over routed experts. Shared experts contribute to `y_{l,t}` but do
not appear in `q_l`, `\mu_l`, or the update map. QED.

## 11. Population Reflected Dynamics

For the router-level population model, freeze the gate, routed experts, shared
experts, and token distribution, and analyze routed pressure dynamics alone.

Let `X \sim D` be a stationary token representation and define the soft
population routing distribution

```math
p_e(X, q)
  = \frac{\exp((a_e(X) + b_e - \gamma_e q_e)/\tau)}
         {\sum_j \exp((a_j(X) + b_j - \gamma_j q_j)/\tau)}.
```

Let

```math
\bar p_e(q) = \mathbb{E}_{X \sim D}[p_e(X, q)].
```

The idealized reflected ODE is

```math
\dot q_e(t)
  =
  \begin{cases}
    \bar p_e(q(t)) - \mu_e, & q_e(t) > 0, \\
    \max(\bar p_e(q(t)) - \mu_e, 0), & q_e(t) = 0.
  \end{cases}
```

This remains a routed-expert-only control system; shared experts are fixed
outside the dual dynamics.

## 12. Convex Potential for the Softmax Population Model

Assume

```math
s_e(X, q) = a_e(X) + b_e - \gamma_e q_e,
\qquad \gamma_e > 0.
```

Define

```math
H(q)
  = \sum_e \gamma_e \mu_e q_e
    + \tau \mathbb{E}_X \left[
        \log \sum_j \exp((a_j(X) + b_j - \gamma_j q_j)/\tau)
      \right].
```

Then

```math
\frac{\partial H}{\partial q_e}
  = \gamma_e (\mu_e - \bar p_e(q)).
```

Thus the unconstrained drift is

```math
\bar p(q) - \mu = - \Gamma^{-1} \nabla H(q),
```

where `\Gamma = diag(\gamma_1, \dots, \gamma_E)`.

The shared experts do not modify this routed-controller potential because they
do not participate in the pressure state.

## 13. Scope Boundary

These results justify the implemented reflected router structure:

- continuous routed feedback,
- drop-aware accepted mass in sparse mode,
- width-scaled reflected updates,
- shared-expert isolation from the pressure controller.

They do not prove:

1. full language-model training convergence,
2. differentiability of sparse top-k boundaries,
3. that shared experts are always beneficial at every scale,
4. that reflected routing outperforms top-1/top-2 in every regime,
5. that accepted sparse feedback exactly matches the idealized population ODE.

Those claims remain empirical.
