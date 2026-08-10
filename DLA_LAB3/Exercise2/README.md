# Exercise 2 — REINFORCE with Variance Reduction on CartPole-v1

## Objective

Exercise 2 extends the REINFORCE implementation developed in Exercise 1.

The environment remains:

```text
CartPole-v1
```

The exercise studies two techniques intended to reduce the variability of the
Monte Carlo policy-gradient update:

1. optional standardization of the episodic returns;
2. a learned state-value baseline.

The goal is not to change the policy architecture or the environment, but to
modify the learning signal used in the REINFORCE loss and compare the resulting
training behaviour.

The Exercise 1 implementation is kept unchanged and is used as the reference
vanilla REINFORCE baseline.

---

## 1. Baseline: vanilla REINFORCE

For each episode, the policy produces a stochastic action distribution

```text
state
  ↓
PolicyNetwork
  ↓
logits
  ↓
Categorical distribution
  ↓
sampled action
```

For every step, the trajectory stores:

```text
log π(a_t | s_t)
reward_t
```

The discounted return is computed backwards as:

```text
G_t = r_t + gamma * G_(t+1)
```

The Exercise 1 policy loss is:

```text
L_policy = - sum_t G_t log π(a_t | s_t)
```

The Monte Carlo return is therefore used directly as the weight of each
log-probability term.

The final Exercise 1 configuration used as reference is:

| Parameter | Value |
|---|---:|
| Environment | CartPole-v1 |
| Policy architecture | 4 → 64 → 2 |
| Activation | ReLU |
| Optimizer | Adam |
| Policy learning rate | 0.005 |
| Discount factor | 0.99 |
| Training episodes | 1000 |
| Evaluation interval | 25 |
| Evaluation episodes | 20 |
| Seed | 42 |
| Evaluation policy | stochastic |

---

## 2. Return standardization

### Motivation

Vanilla REINFORCE uses the raw Monte Carlo returns directly.

The numerical scale of the returns may vary considerably during training, which
also changes the scale of the policy-gradient updates.

Exercise 2 therefore makes episodic return standardization optional.

For the returns of a single episode:

```text
G_0, G_1, ..., G_(T-1)
```

the mean is:

```text
mu_G = mean(G)
```

and the population standard deviation is:

```text
sigma_G = std(G)
```

The standardized return is:

```text
G_hat_t = (G_t - mu_G) / (sigma_G + epsilon)
```

with:

```text
epsilon = 1e-8
```

for numerical stability.

The implementation uses:

```python
returns.std(correction=0)
```

so that the complete vector of episodic returns is treated as the population
being normalized.

---

### Effect on the policy loss

Without standardization:

```text
L_policy = - sum_t G_t log π(a_t | s_t)
```

With standardization:

```text
L_policy = - sum_t G_hat_t log π(a_t | s_t)
```

Nothing changes in the PolicyNetwork.

Only the coefficient multiplying each `log_prob` changes.

The standardized returns do not require gradients:

```text
standardized_returns.requires_grad = False
```

while the stored log-probabilities remain connected to the PolicyNetwork:

```text
log_prob.requires_grad = True
```

Therefore the gradient flow remains:

```text
policy loss
    ↓
log_prob
    ↓
logits
    ↓
PolicyNetwork parameters
```

---

### Numerical verification

A simple test was performed using:

```text
returns = [10, 8, 6, 4]
```

The result was approximately:

```text
[ 1.3416,  0.4472, -0.4472, -1.3416]
```

with:

```text
mean ≈ 0
std  ≈ 1
```

A zero-variance test:

```text
[5, 5, 5, 5]
```

produced:

```text
[0, 0, 0, 0]
```

without NaN or Inf values.

---

## 3. Learned value baseline

### State-value function

The second part introduces an additional neural network:

```text
V_w(s)
```

which approximates the state-value function:

```text
v_pi(s) = E_pi[G_t | S_t = s]
```

The PolicyNetwork and ValueNetwork solve different problems.

#### PolicyNetwork

```text
input:  state s_t
output: two action logits
role:   define π_theta(a | s)
```

Architecture:

```text
4 → 64 → 2
```

#### ValueNetwork

```text
input:  state s_t
output: scalar state value
role:   estimate expected future return
```

Architecture:

```text
4 → 64 → 1
```

with ReLU activation in the hidden layer.

No Softmax is used in the ValueNetwork because the output is a scalar value,
not a probability.

---

## 4. Extended trajectory

Exercise 1 mainly stores:

```text
log_probs
rewards
```

The value baseline additionally requires the states.

For each step:

```text
state_t
log_prob_t
reward_t
```

are stored.

For an episode of length `T`:

```text
states      shape: (T, 4)
log_probs   shape: (T,)
returns     shape: (T,)
values      shape: (T,)
advantages  shape: (T,)
```

The state associated with an action is stored before calling `env.step()`:

```text
S_t
  ↓
PolicyNetwork
  ↓
A_t
  ↓
env.step(A_t)
```

This guarantees that:

```text
states[t]
log_probs[t]
returns[t]
```

all refer to the same time step.

---

## 5. Advantage estimate

Instead of using the raw return directly, the policy uses:

```text
A_t ≈ G_t - V_w(S_t)
```

This can be interpreted as comparing:

```text
actual return
      -
expected return from the current state
```

If:

```text
A_t > 0
```

the observed return was better than expected from that state.

If:

```text
A_t < 0
```

the observed return was worse than expected.

The policy loss becomes:

```text
L_policy =
    - sum_t (G_t - V_w(S_t)) log π_theta(A_t | S_t)
```

---

## 6. Gradient flow and detach

The ValueNetwork must not be updated through the policy loss.

The implementation therefore uses:

```python
advantages = returns - values.detach()
```

instead of:

```python
advantages = returns - values
```

Without `detach()`:

```text
policy loss
    ↓
advantage
    ↓
ValueNetwork
```

would create an unintended gradient path.

With `detach()`:

```text
advantages.requires_grad = False
```

and the policy gradient is restricted to:

```text
POLICY

policy loss
    ↓
log_prob
    ↓
Categorical distribution
    ↓
logits
    ↓
PolicyNetwork parameters
```

The ValueNetwork is trained independently:

```text
VALUE

value loss
    ↓
V_w(S_t)
    ↓
ValueNetwork parameters
```

A dedicated gradient-flow test verified that:

```text
after policy_loss.backward():

PolicyNetwork gradients     → True
ValueNetwork gradients      → False
```

and:

```text
after value_loss.backward():

PolicyNetwork gradients     → False
ValueNetwork gradients      → True
```

No `retain_graph=True` is required.

---

## 7. Value-network target and loss

The Monte Carlo return is used as the target:

```text
target = G_t
```

The value loss is the mean squared error:

```text
L_value = MSE(V_w(S_t), G_t)
```

or:

```text
L_value =
    mean_t (V_w(S_t) - G_t)^2
```

The returns do not require gradients.

The value predictions do require gradients because they depend on the
ValueNetwork parameters.

Two independent Adam optimizers are used:

```text
policy optimizer → PolicyNetwork
value optimizer  → ValueNetwork
```

---

## 8. Verification of the ValueNetwork

Before integrating the network into the full reinforcement-learning loop, several
isolated tests were performed.

### Shape test

Single state:

```text
input:  (4,)
output: scalar
```

Trajectory:

```text
input:  (T, 4)
output: (T,)
```

The output shape is directly compatible with the Monte Carlo return vector.

### Fixed-target learning test

A fixed CartPole trajectory of length 20 was collected.

Its returns ranged approximately from:

```text
18.21
...
1.00
```

The initially untrained ValueNetwork produced values around zero.

The same fixed trajectory was then used for 200 ValueNetwork updates.

The MSE decreased from:

```text
126.54
```

to:

```text
6.49
```

confirming that the network, target, loss and optimization path were functioning
correctly when the target was stationary.

### Online diagnostic

A 200-episode preliminary training run with the learned baseline was followed by
evaluation on 20 fresh trajectories.

The collected diagnostic contained:

```text
9817 states
```

The statistics were:

```text
Mean return:           79.98
Mean predicted value:  75.77
Mean advantage:         4.22
Std advantage:         23.54
```

The learned baseline obtained:

```text
MSE = 571.93
```

while the zero baseline:

```text
V(s) = 0
```

obtained:

```text
MSE = 7000.06
```

The learned ValueNetwork therefore captured useful information about the scale of
the returns.

Since the RL target changes together with the policy and visited states, the
per-episode value loss is not expected to decrease monotonically during the full
training run.

---

## 9. Experimental protocol

Three configurations were compared.

### Configuration 1 — Vanilla REINFORCE

```text
standardize_returns = False
value_baseline      = False
```

### Configuration 2 — Standardized returns

```text
standardize_returns = True
value_baseline      = False
```

### Configuration 3 — Learned value baseline

```text
standardize_returns = False
value_baseline      = True
```

The remaining policy and evaluation parameters were kept fixed as much as
possible:

| Parameter | Value |
|---|---:|
| Environment | CartPole-v1 |
| Policy | 4 → 64 → 2 |
| Policy activation | ReLU |
| Policy optimizer | Adam |
| Policy learning rate | 0.005 |
| Gamma | 0.99 |
| Training episodes | 1000 |
| Evaluation interval | 25 |
| Evaluation episodes | 20 |
| Seed | 42 |
| Evaluation | stochastic |

For the learned baseline:

| Parameter | Value |
|---|---:|
| ValueNetwork | 4 → 64 → 1 |
| Activation | ReLU |
| Optimizer | Adam |
| Value learning rate | 0.005 |
| Target | Monte Carlo return |
| Loss | MSE |

Only one seed was used for this first controlled comparison.

The results must therefore be interpreted as behaviour observed for `seed=42`,
not as a statistically complete ranking of the algorithms.

---

## 10. Results

The complete comparison uses the 40 periodic evaluations generated during each
1000-episode training run.

| Configuration | Mean evaluation reward | First evaluation at 500 | Evaluations at 500 | Evaluations ≥ 475 | Mean of final 10 evaluations |
|---|---:|---:|---:|---:|---:|
| Vanilla REINFORCE | 342.09 | 400 | 12 / 40 | 16 / 40 | 405.52 |
| Standardized returns | 376.30 | 225 | 13 / 40 | 24 / 40 | 453.78 |
| Value baseline | **449.82** | 250 | **23 / 40** | **30 / 40** | **500.00** |

The mean reward during the first ten evaluations was:

```text
Vanilla REINFORCE:     155.57
Standardized returns:  252.87
Value baseline:        319.96
```

### Vanilla REINFORCE

The vanilla policy learns the task and repeatedly reaches the maximum reward,
but the training remains highly unstable.

For example:

```text
episode 725 → 500.00
episode 750 → 306.40
episode 775 → 150.45
episode 800 → 128.95
```

The policy eventually recovers and the final six evaluations reach 500, but the
run clearly demonstrates the high variability of the raw Monte Carlo update.

### Standardized returns

Return standardization improves the initial learning speed.

The first exact evaluation reward of 500 occurs at episode:

```text
225
```

compared with episode 400 for vanilla REINFORCE.

However, standardization does not remove the instability.

A strong degradation occurs around episodes 600–675:

```text
600 → 239.65
625 → 134.50
650 → 125.60
675 → 119.95
```

and another isolated collapse occurs at episode 950:

```text
925 → 500.00
950 → 130.35
975 → 476.00
```

The technique therefore controls the scale of the return signal and improves
the average training behaviour, but does not provide a state-dependent
expectation.

### Learned value baseline

The learned baseline gives the strongest behaviour in this single-seed
experiment.

The policy already reaches:

```text
episode 125 → 431.15
episode 150 → 447.35
episode 175 → 493.45
```

and reaches the first exact reward of 500 at episode 250.

The important difference is not only the first maximum reward, but the subsequent
stability.

After the policy enters the high-reward region, the drops are much smaller than
those observed with the other configurations.

The final part of the run is:

```text
725  → 500.00
750  → 500.00
775  → 500.00
800  → 500.00
825  → 500.00
850  → 500.00
875  → 500.00
900  → 500.00
925  → 500.00
950  → 500.00
975  → 500.00
1000 → 500.00
```

The final 12 consecutive evaluations therefore remain at the maximum CartPole
reward.

---

## 11. Interpretation

The three experiments illustrate progressively stronger variance-reduction
mechanisms.

### Vanilla REINFORCE

Uses:

```text
G_t
```

directly.

It can solve CartPole, but the Monte Carlo update shows strong variability and
can destroy a previously high-performing policy.

### Return standardization

Uses:

```text
(G_t - mean(G)) / (std(G) + epsilon)
```

The signal is centered and its scale is controlled.

In the observed run this produces faster initial learning and higher average
performance than vanilla REINFORCE.

However, the transformation is based only on statistics of the current episode
and contains no state-specific information.

### Value baseline

Uses:

```text
G_t - V_w(S_t)
```

The update now measures whether the obtained return was better or worse than the
return expected from the current state.

For `seed=42`, this results in the highest average evaluation reward and the most
stable final policy.

This behaviour is consistent with the theoretical purpose of a learned baseline:
reduce the variability of the policy-gradient estimator without changing the
expected policy gradient when the baseline is independent of the selected action.

---

## 12. Value-loss interpretation

The value loss is not monotonic during RL training.

For example, values observed during the final run include:

```text
episode 25   → 282.53
episode 100  → 2954.41
episode 250  → 516.68
episode 700  → 355.38
episode 1000 → 568.24
```

This does not by itself indicate divergence.

The ValueNetwork is solving a non-stationary regression problem because the
PolicyNetwork changes continuously.

As the policy changes, so do:

```text
visited states
episode lengths
Monte Carlo returns
target distribution
```

The fixed-trajectory test demonstrated that the same ValueNetwork can strongly
reduce the MSE when the target is held constant.

---

## 13. Results plot

The final comparison is stored in:

```text
plots/evaluation_comparison.png
```

The plot shows:

- slower and highly variable vanilla REINFORCE;
- faster learning but remaining collapses with standardized returns;
- rapid entry into the high-reward region and substantially greater stability
  with the learned value baseline.

The horizontal dashed line represents the maximum CartPole-v1 reward of 500.

---

## 14. Project structure

```text
DLA_LAB3/
├── models.py
├── reinforce.py
│
├── Exercise1/
│   └── ...
│
└── Exercise2/
    ├── main.py
    ├── reinforce_ex2.py
    ├── value_baseline_main.py
    ├── plot_results.py
    ├── README.md
    │
    ├── plots/
    │   └── evaluation_comparison.png
    │
    └── runs/
        ├── no_standardization_seed42/
        │   ├── config.json
        │   ├── training_metrics.csv
        │   ├── evaluation_metrics.csv
        │   └── policy.pt
        │
        ├── standardized_returns_seed42/
        │   ├── config.json
        │   ├── training_metrics.csv
        │   ├── evaluation_metrics.csv
        │   └── policy.pt
        │
        └── value_baseline_seed42/
            ├── config.json
            ├── training_metrics.csv
            ├── evaluation_metrics.csv
            ├── policy.pt
            └── value.pt
```

`models.py` contains both the existing `PolicyNetwork` and the new
`ValueNetwork`.

The original `reinforce.py` remains unchanged and continues to support Exercise
1.

`Exercise2/reinforce_ex2.py` contains the Exercise 2 trajectory collection,
optional return standardization, learned value-baseline update and training
loops.

`Exercise2/main.py` runs the comparison between raw and standardized returns.

`Exercise2/value_baseline_main.py` runs the learned value-baseline experiment.

`Exercise2/plot_results.py` produces the final comparison figure.

---

## 15. Running Exercise 2

From:

```text
DLA_LAB3/
```

activate the environment:

```bash
conda activate DRL
```

### Return-standardization comparison

```bash
python -m Exercise2.main
```

This executes:

```text
standardization OFF
standardization ON
```

using the same experimental configuration.

### Value baseline

```bash
python -m Exercise2.value_baseline_main
```

### Generate the comparison plot

```bash
python -m Exercise2.plot_results
```

---

## 16. Main conclusion

Exercise 2 demonstrates that the learning signal used by REINFORCE has a major
effect on training behaviour.

Episodic return standardization improves the initial learning speed and average
performance compared with raw Monte Carlo returns, but does not eliminate large
policy collapses.

The learned state-value baseline produces the best behaviour in the controlled
single-seed experiment. It reaches high reward rapidly and maintains the solved
CartPole policy much more consistently during the second half of training.

The result supports the role of a learned baseline as a variance-reduction
mechanism and provides a direct practical interpretation of the advantage term:

```text
A_t ≈ G_t - V_w(S_t)
```

The comparison is currently based on a single seed. A multi-seed experiment
could be used as an optional extension to quantify the robustness of these
differences, but it is not required for the core implementation of Exercise 2.
