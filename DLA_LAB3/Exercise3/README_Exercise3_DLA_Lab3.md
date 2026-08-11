# DLA Lab 3 — Exercise 3
## Deep Q-Learning on CartPole-v1 and LunarLander-v3

## 1. Objective

This exercise implements and evaluates a Deep Q-Network (DQN) agent on two discrete-control environments:

- `CartPole-v1`
- `LunarLander-v3`

The goal is to move from the policy-gradient methods studied in Exercise 1 and Exercise 2 to a value-based, off-policy reinforcement learning method.

The implementation includes the two central stabilizing components required by DQN:

- Experience Replay through a Replay Buffer;
- a separate, slowly updated Target Q-Network.

The same generic DQN implementation is used for both environments.

---

# 2. From REINFORCE to DQN

Exercise 1 and Exercise 2 used policy-gradient methods.

The policy network directly represented

\[
\pi_\theta(a|s)
\]

and actions were sampled from a categorical distribution.

DQN follows a different approach.

Instead of representing the policy explicitly, the neural network approximates the action-value function

\[
Q_\theta(s,a)
\]

which estimates the expected discounted return obtained by taking action \(a\) in state \(s\) and subsequently behaving according to the learned policy.

For a state \(s\), the network outputs one Q-value for each possible action:

\[
Q_\theta(s)
=
[
Q_\theta(s,a_1),
Q_\theta(s,a_2),
\dots
]
\]

The greedy action is therefore

\[
a^*
=
\arg\max_a Q_\theta(s,a)
\]

Unlike the policy network used by REINFORCE, the output layer of the Q-network does not use Softmax.

The outputs are raw Q-value estimates, not probabilities.

---

# 3. Environments

## 3.1 CartPole-v1

Observation dimension:

```text
4
```

Number of discrete actions:

```text
2
```

The Q-network therefore has the architecture:

```text
4 → 64 → 2
```

with ReLU activation.

---

## 3.2 LunarLander-v3

Observation dimension:

```text
8
```

Number of discrete actions:

```text
4
```

The same generic `QNetwork` becomes:

```text
8 → 64 → 4
```

This demonstrates that the DQN implementation itself is environment-independent for discrete environments with vector observations.

---

# 4. Q-Network

The Q-network is implemented in:

```text
Exercise3/dqn.py
```

Its structure is:

```text
state
  ↓
Linear(state_dim, 64)
  ↓
ReLU
  ↓
Linear(64, action_dim)
  ↓
Q-values
```

For a minibatch of size \(B\):

CartPole:

```text
states            (B, 4)
Q(states)         (B, 2)
```

LunarLander:

```text
states            (B, 8)
Q(states)         (B, 4)
```

Only the Q-value associated with the action that was actually executed is used in the loss.

Given

```text
Q(states) = (B, A)
actions   = (B,)
```

the selected values are

```text
Q(s,a) = (B,)
```

using a gather operation.

---

# 5. Temporal-Difference Learning

DQN does not wait for the end of an episode before computing the learning target.

For a non-terminal transition

\[
(s_t,a_t,r_t,s_{t+1})
\]

the Temporal-Difference target is

\[
y_t
=
r_t
+
\gamma
\max_a
Q_{\text{target}}(s_{t+1},a)
\]

For a terminal transition:

\[
y_t = r_t
\]

The complete target used in the implementation is

\[
y_t
=
r_t
+
\gamma
(1-\text{terminated}_t)
\max_a
Q_{\text{target}}(s_{t+1},a)
\]

This is a bootstrapped target because part of the target is itself estimated by a neural network.

This differs from REINFORCE, where Monte Carlo returns are computed using actual future rewards from a complete trajectory.

---

# 6. `terminated` and `truncated`

Gymnasium distinguishes between:

```text
terminated
```

and

```text
truncated
```

Both conditions stop the current rollout:

```python
terminated or truncated
```

However, they are not treated identically in the TD target.

A true terminal state prevents bootstrapping.

A time-limit truncation does not necessarily represent a terminal state of the underlying Markov Decision Process.

For this reason the Replay Buffer stores both values separately.

The bootstrap mask uses `terminated`.

---

# 7. Replay Buffer

DQN is an off-policy algorithm.

Transitions generated while interacting with the environment are stored as:

```text
(
    state,
    action,
    reward,
    next_state,
    terminated,
    truncated
)
```

The Replay Buffer provides two important benefits.

## 7.1 Experience reuse

A transition can participate in several optimization steps instead of being discarded immediately.

## 7.2 Reduced temporal correlation

Sequential environment transitions are strongly correlated.

Random minibatch sampling from a Replay Buffer provides a less correlated training distribution.

For a batch of size \(B\):

```text
states        (B, state_dim) float32
actions       (B,)           int64
rewards       (B,)           float32
next_states   (B, state_dim) float32
terminated    (B,)           bool
truncated     (B,)           bool
```

Experience stored in the Replay Buffer is detached from the PyTorch computational graph.

---

# 8. Online Network and Target Network

DQN uses two Q-networks with the same architecture:

```text
online_network
target_network
```

The online network is optimized through gradient descent.

The target network is used only to compute the TD target.

The target branch is evaluated without gradient tracking:

```text
next_state
    ↓
target network
    ↓
max Q(next_state)
    ↓
TD target
```

No optimizer is associated with the target network.

Instead, its parameters are periodically copied from the online network:

```text
target ← online
```

This project uses a hard periodic target update.

The purpose is to avoid constructing the TD target with the same rapidly changing network being optimized.

---

# 9. Gradient Flow

The intended gradient flow is:

```text
states
   ↓
online Q-network
   ↓
Q(s,a)
   ↓
loss
   ↓
backward
   ↓
online network parameters
```

The target branch does not receive gradients:

```text
next_states
   ↓
target Q-network
   ↓
TD target

NO backward
```

This property was explicitly verified through isolated gradient-flow tests before integrating the full training loop.

---

# 10. Exploration with epsilon-greedy

During training the behavior policy uses epsilon-greedy exploration.

With probability \(\epsilon\):

```text
random action
```

otherwise:

\[
a
=
\arg\max_a Q(s,a)
\]

A linear epsilon schedule is used:

```text
epsilon_start → epsilon_end
```

as a function of environment steps.

Evaluation is always greedy:

```text
epsilon = 0
```

and performs:

- no Replay Buffer writes;
- no optimizer updates;
- no gradient computation.

---

# 11. Loss functions studied

Two losses were compared on CartPole.

## Mean Squared Error

\[
L =
\frac{1}{B}
\sum_i
(Q(s_i,a_i)-y_i)^2
\]

## Huber / Smooth L1

Huber loss was also evaluated as a potentially more robust alternative in the presence of large TD errors.

In the experiments performed here, MSE produced better policies than Huber under the tested configuration.

---

# 12. Validation methodology

The implementation was developed incrementally.

Before performing complete training runs, the following isolated tests were executed.

### QNetwork tests

- CartPole single-state shape:
  `(4,) → (2,)`
- CartPole batch shape:
  `(B,4) → (B,2)`
- LunarLander single-state shape:
  `(8,) → (4,)`
- LunarLander batch shape:
  `(B,8) → (B,4)`
- output gradient availability

### Replay Buffer tests

- insertion;
- capacity management;
- random sampling;
- CartPole shapes;
- LunarLander shapes;
- tensor dtypes;
- detached stored experience;
- distinction between `terminated` and `truncated`;
- invalid batch-size handling.

### TD-learning tests

- exact numerical TD target;
- non-terminal bootstrap;
- terminal no-bootstrap case;
- Q-value action selection through `gather`;
- finite loss;
- online-network gradient presence;
- target-network gradient absence.

### Target network tests

- initial synchronization;
- exact parameter equality after hard synchronization.

### Integration tests

A complete end-to-end CartPole smoke test verified:

```text
environment
→ epsilon-greedy
→ Replay Buffer
→ minibatch
→ TD loss
→ backward
→ optimizer step
→ target synchronization
```

A separate LunarLander smoke test verified that the same implementation works with an 8-dimensional state and 4 actions.

---

# 13. CartPole experiments

Three controlled configurations were studied.

## Experiment A

```text
Environment:             CartPole-v1
Episodes:                250
Gamma:                   0.99
Optimizer:               Adam
Learning rate:           1e-3
Loss:                    MSE
Hidden dimension:        64
Replay capacity:         10,000
Batch size:              64
Replay warm-up:          500
epsilon start:           1.0
epsilon end:             0.05
epsilon decay:           10,000 steps
Target sync:             every 250 updates
Training seed:           42
```

Training evaluation:

```text
Best evaluation reward: 303.10
Final evaluation reward: 238.40
```

Robust evaluation of the final checkpoint over 100 episodes:

```text
Mean reward:   230.94
Std reward:     35.69
Median reward: 219.00
Min:           180
Max:           386
```

---

## Experiment B

The learning rate was reduced while keeping the other experimental choices fixed.

```text
Learning rate: 5e-4
Loss:          MSE
```

Training evaluation:

```text
Best evaluation reward:
357.50 at episode 200

Final evaluation reward:
239.70
```

Robust evaluation of the best checkpoint:

```text
Mean reward:   280.84
Std reward:     75.88
Median reward: 258.50
Min:           185
Max:           500
```

Robust evaluation of the final checkpoint:

```text
Mean reward:   238.68
Std reward:     53.85
Median reward: 216.50
Min:           174
Max:           461
```

The lower learning rate therefore produced the best CartPole checkpoint observed in the experiments.

---

## Experiment C

The third experiment kept:

```text
learning rate = 5e-4
```

but replaced MSE with Huber loss.

Robust evaluation:

### training-best checkpoint

```text
Mean reward:   169.60
Std reward:     52.17
Median reward: 152.00
```

### final checkpoint

```text
Mean reward:   187.99
Std reward:     48.16
Median reward: 172.50
```

Huber loss did not improve the agent under the tested configuration.

---

# 14. Selected CartPole model

The selected CartPole checkpoint is:

```text
runs/cartpole_dqn_lr0.0005_seed42/best_q_network.pt
```

Robust evaluation over 100 greedy episodes using seeds `1000–1099`:

```text
Mean reward:        280.84
Std reward:          75.88
Median reward:      258.50
Minimum reward:     185
Maximum reward:     500

Episodes >= 100:    100%
Episodes >= 200:     95%
```

---

# 15. LunarLander experiments

The initial 500-episode pilot used:

```text
Environment:             LunarLander-v3
Gamma:                   0.99
Optimizer:               Adam
Learning rate:           5e-4
Loss:                    MSE
Hidden dimension:        64

Replay capacity:         50,000
Batch size:              64
Replay warm-up:          1,000

epsilon start:           1.0
epsilon end:             0.05
epsilon decay:           50,000 steps

Target synchronization:  every 500 updates
Training seed:           42
```

The evaluation reward progressed approximately from:

```text
episode 25      -288.61
episode 100     -220.97
episode 175     -137.23
episode 250     -103.76
episode 275      -33.78
episode 300       -0.87
episode 400       70.16
episode 500      157.63
```

The continuous improvement at the end of the pilot motivated a larger training budget.

---

# 16. LunarLander 1000-episode run

The same algorithm and hyperparameters were used for 1000 episodes.

Only the training budget was increased.

Selected periodic evaluations include:

```text
episode 500      157.63
episode 550      160.51
episode 575      198.33
episode 650      165.97
episode 725      165.29
episode 750      154.54
episode 850      130.77
episode 950      145.82
episode 975      165.23
episode 1000     135.83
```

The evaluation curve demonstrates substantial learning but also considerable DQN instability.

---

# 17. Robust LunarLander evaluation

All final model comparisons were performed using:

```text
100 greedy episodes
seeds 1000–1099
```

## 500-episode final checkpoint

```text
Mean reward:            93.91
Std reward:            135.19
Median reward:         137.55
Minimum reward:       -319.46
Maximum reward:        272.81

Positive episodes:      79%
Reward >= 100:          74%
Reward >= 200:          14%
```

---

## 1000-episode training-best checkpoint

The periodic evaluation during training selected the episode-575 model as the best checkpoint because it obtained an average reward of approximately 198 over 10 evaluation episodes.

However, robust evaluation over 100 episodes produced:

```text
Mean reward:            31.03
Std reward:            254.19
Median reward:         132.00
Minimum reward:       -807.30
Maximum reward:        305.01

Positive episodes:      61%
Reward >= 100:          55%
Reward >= 200:          33%
```

This result demonstrates that checkpoint selection based on a small evaluation sample can be noisy.

---

## 1000-episode final checkpoint

Robust evaluation of the final model produced:

```text
Mean reward:           172.68
Std reward:             70.02
Median reward:         184.82
Minimum reward:       -101.21
Maximum reward:        282.87

Positive episodes:      96%
Reward >= 100:          84%
Reward >= 200:          32%
```

This checkpoint produced the best overall balance between average reward, median reward and robustness.

---

# 18. Selected LunarLander model

The selected LunarLander model is:

```text
runs/lunarlander_dqn_final_1000ep_seed42/
final_q_network.pt
```

It was selected using the independent robust evaluation rather than only the periodic training evaluation.

This distinction is important.

The checkpoint with the highest 10-episode periodic evaluation was not the most reliable checkpoint over 100 independently seeded episodes.

---

# 19. TD-loss analysis

The LunarLander TD loss provides additional information about the learning dynamics.

For the 500-episode pilot:

```text
First 100 updated episodes:
mean TD loss ≈ 59.40

Middle 100 episodes:
mean TD loss ≈ 17.38

Last 100 episodes:
mean TD loss ≈ 11.06
```

At the same time, mean training reward improved from negative values to positive values.

The complete 1000-episode curve shows that TD loss is not monotonic.

After the large initial decrease, new peaks appear as the policy changes and the distribution of replay-buffer experience evolves.

Therefore TD loss should not be interpreted exactly like a conventional supervised-learning validation loss.

The primary metric remains environment return.

---

# 20. Main experimental conclusion

The experiments demonstrate several important DQN properties.

### Replay and Target Networks enable learning

The same DQN implementation successfully learns non-trivial policies on both CartPole and LunarLander.

### Hyperparameters affect stability

On CartPole:

```text
MSE + lr=5e-4
```

performed better than:

```text
MSE + lr=1e-3
```

and

```text
Huber + lr=5e-4
```

under the tested configurations.

### More training is not monotonically better

Both CartPole and LunarLander demonstrate oscillations in evaluation reward during training.

Deep Q-Learning optimizes a moving bootstrapped target while the behavior policy and Replay Buffer distribution are also evolving.

### Small evaluation sets can misidentify the best model

This was particularly clear for LunarLander.

The checkpoint selected as best using 10 evaluation episodes was substantially worse than the final checkpoint when both were evaluated over the same 100 seeds.

A larger independent evaluation is therefore used for final model selection.

---

# 21. Plots

Generated plots are stored in:

```text
Exercise3/plots/
```

## CartPole

```text
cartpole_evaluation_comparison.png
cartpole_robust_comparison.png
cartpole_selected_reward_distribution.png
```

## LunarLander

```text
lunarlander_evaluation_curve.png
lunarlander_training_reward.png
lunarlander_td_loss.png
lunarlander_robust_comparison.png
lunarlander_selected_reward_distribution.png
```

In robust-comparison figures, error bars represent the standard deviation of episode returns.

They are not confidence intervals and are not standard errors of the mean.

---

# 22. Reproducible evaluation

Robust evaluation is implemented in:

```text
Exercise3/evaluate_results.py
```

It evaluates all relevant checkpoints using:

```text
100 episodes
seeds 1000–1099
greedy policy
```

Run:

```bash
python -m Exercise3.evaluate_results
```

Outputs:

```text
Exercise3/results/
├── robust_evaluation_episodes.csv
└── robust_evaluation_summary.csv
```

`robust_evaluation_episodes.csv` contains one row per evaluation episode.

`robust_evaluation_summary.csv` contains aggregate statistics for every checkpoint.

---

# 23. Reproducing plots

Plots are generated exclusively from persisted CSV files.

No training or environment evaluation is repeated by the plotting script.

Run:

```bash
python -m Exercise3.plot_results
```

Outputs are written to:

```text
Exercise3/plots/
```

---

# 24. Running CartPole training

From the `DLA_LAB3` directory:

```bash
python -m Exercise3.main
```

The current `main.py` configuration reproduces the selected CartPole configuration using MSE loss and learning rate 5e-4. A dedicated run name is used so that previously generated experimental artifacts are not overwritten.

Previously generated runs are preserved under `Exercise3/runs/`.

---

# 25. Running LunarLander training

From the `DLA_LAB3` directory:

```bash
python -m Exercise3.lunarlander_main
```

The current LunarLander configuration runs the final 1000-episode experiment.

---

# 26. Project structure

```text
Exercise3/
├── dqn.py
├── main.py
├── lunarlander_main.py
├── evaluate_results.py
├── plot_results.py
├── README_Exercise3_DLA_Lab3.md
│
├── runs/
│   ├── cartpole_dqn_pilot_seed42/
│   ├── cartpole_dqn_lr0.0005_seed42/
│   ├── cartpole_dqn_huber_lr0.0005_seed42/
│   ├── lunarlander_dqn_pilot_seed42/
│   └── lunarlander_dqn_final_1000ep_seed42/
│
├── results/
│   ├── robust_evaluation_episodes.csv
│   └── robust_evaluation_summary.csv
│
└── plots/
    ├── cartpole_evaluation_comparison.png
    ├── cartpole_robust_comparison.png
    ├── cartpole_selected_reward_distribution.png
    ├── lunarlander_evaluation_curve.png
    ├── lunarlander_training_reward.png
    ├── lunarlander_td_loss.png
    ├── lunarlander_robust_comparison.png
    └── lunarlander_selected_reward_distribution.png
```

---

# 27. Main files

## `dqn.py`

Contains the generic DQN implementation:

- `QNetwork`
- `ReplayBuffer`
- TD-loss computation
- epsilon-greedy action selection
- target-network synchronization
- epsilon schedule
- greedy evaluation
- DQN training loop

## `main.py`

CartPole experimental entry point.

## `lunarlander_main.py`

LunarLander experimental entry point.

## `evaluate_results.py`

Independent robust checkpoint evaluation.

## `plot_results.py`

Generation of all final figures from persisted CSV artifacts.

---

# 28. Limitations

The experiments use a single training seed (`42`).

The final robust evaluations use 100 independently seeded environment episodes, which improves confidence in policy evaluation but does not measure variability across independently trained agents.

A stronger experimental protocol could train several agents using different training seeds and report mean and variance across runs.

The implementation also uses the original DQN target:

\[
\max_a Q_{\text{target}}(s',a)
\]

which is subject to the maximization bias addressed by Double DQN.

Other extensions not explored here include:

- Double DQN;
- Dueling DQN;
- Prioritized Experience Replay;
- soft target updates;
- multi-step returns.

These were intentionally excluded in order to keep the implementation focused on the core DQN algorithm requested by the exercise.

---

# 29. Final result

The exercise successfully implements Deep Q-Learning with:

```text
Q-network
Replay Buffer
epsilon-greedy exploration
bootstrapped TD targets
Target Q-Network
hard target synchronization
greedy evaluation
robust checkpoint evaluation
```

and applies the same implementation to both:

```text
CartPole-v1
LunarLander-v3
```

Selected robust results:

```text
CartPole-v1
Mean reward: 280.84
100 evaluation episodes

LunarLander-v3
Mean reward: 172.68
Median reward: 184.82
Positive episodes: 96%
100 evaluation episodes
```

The experiments also highlight a central practical property of Deep Q-Learning: strong performance can coexist with substantial training instability, making independent and sufficiently large evaluation sets important for meaningful model selection.
