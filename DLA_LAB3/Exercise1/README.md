# Exercise 1 — REINFORCE on CartPole-v1

## Objective

The goal of Exercise 1 is to implement and evaluate a simple REINFORCE agent on
`CartPole-v1`.

The exercise follows three main steps:

1. understand the Gymnasium CartPole environment;
2. implement a working REINFORCE policy-gradient agent;
3. improve the evaluation procedure by periodically evaluating the policy on
   separate episodes without updating the network.

The implementation uses normal Python modules rather than keeping the complete
algorithm inside the assignment notebook.

---

## Environment

The environment is:

```text
CartPole-v1
```

The observation contains four continuous values:

```text
[x, x_dot, theta, theta_dot]
```

corresponding to:

- cart position;
- cart velocity;
- pole angle;
- pole angular velocity.

The action space is:

```text
Discrete(2)
```

The current Gymnasium API is used:

```python
observation, info = env.reset()

next_observation, reward, terminated, truncated, info = env.step(action)
```

An episode stops when:

```python
terminated or truncated
```

`terminated` and `truncated` are kept distinct.

CartPole gives a reward of `+1` for every executed step and the `v1`
environment has a maximum episode length of 500 steps.

---

## Policy Network

The policy is implemented in `models.py`.

Architecture:

```text
4 inputs
   ↓
Linear(4, 64)
   ↓
ReLU
   ↓
Linear(64, 2)
   ↓
2 logits
```

The two outputs are logits associated with the two possible actions.

A categorical distribution is constructed directly from the logits:

```python
distribution = Categorical(logits=logits)
```

During training, the action is sampled from this distribution.

---

## REINFORCE

For each episode, the agent stores:

```text
log π(a_t | s_t)
reward_t
```

After the episode finishes, discounted returns are computed backwards:

```text
G_t = r_t + gamma * G_(t+1)
```

The policy loss is:

```text
L = - sum_t G_t log π(a_t | s_t)
```

The negative sign is required because PyTorch minimizes the loss, while
REINFORCE aims to maximize expected return.

The update is performed once per complete episode:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

The Gymnasium environment is outside the PyTorch computational graph.
Gradients flow through the stored `log_prob` tensors back to the policy network.

---

## Evaluation

Training and evaluation are separated.

Every `N` training episodes, the current policy is evaluated for `M` complete
episodes without updating its parameters.

The evaluation uses:

```python
torch.inference_mode()
```

and no optimizer or backward pass is executed.

The collected metrics are:

- average total reward;
- average episode length.

For this implementation:

```text
N = 25
M = 20
```

Evaluation actions are sampled from the learned stochastic policy.

For CartPole, average reward and average episode length are numerically equal
because each executed step gives reward `+1`. Both metrics are nevertheless
stored separately.

---

## Experimental configuration

The final configuration is:

| Parameter | Value |
|---|---:|
| Environment | CartPole-v1 |
| Policy | 4 → 64 → 2 |
| Activation | ReLU |
| Optimizer | Adam |
| Learning rate | 0.005 |
| Discount factor | 0.99 |
| Training episodes | 1000 |
| Evaluation interval | 25 |
| Evaluation episodes | 20 |
| Seed | 42 |

A preliminary run used learning rate `0.01` while keeping the remaining
configuration unchanged.

---

## Learning-rate comparison

Two configurations were evaluated with the same seed and experimental protocol.

### Adam, learning rate = 0.01

The policy learned useful behaviour and reached the maximum evaluation reward
multiple times, but training was highly unstable.

Examples from the evaluation history include:

```text
episode 550  → 500.00
episode 600  → 500.00
episode 675  → 500.00
episode 700  → 500.00
episode 825  →  99.25
episode 850  →  82.30
episode 1000 →  78.80
```

The run therefore demonstrates that vanilla REINFORCE can learn the task, but
the chosen update size does not preserve the high-performing policy reliably.

### Adam, learning rate = 0.005

Reducing the learning rate produced a substantially more stable result.

The policy reached the maximum evaluation score repeatedly and the final six
evaluation checkpoints were:

```text
episode 875  → 500.00
episode 900  → 500.00
episode 925  → 500.00
episode 950  → 500.00
episode 975  → 500.00
episode 1000 → 500.00
```

The training is still not monotonic. For example, the evaluation reward falls
between episodes 750 and 800 before recovering.

This behaviour is consistent with the high variance of vanilla Monte Carlo
policy-gradient updates.

No return standardization or value baseline is introduced in Exercise 1,
because these modifications are studied separately in Exercise 2.

---

## Final policy verification

The final policy was saved to:

```text
runs/lr0.005_seed42_final/policy.pt
```

The checkpoint was loaded into a newly instantiated `PolicyNetwork` and
evaluated again on 20 stochastic episodes.

The loaded policy obtained:

```text
Average reward:         491.15
Average episode length: 491.15
```

A separate visual test using:

```python
gym.make("CartPole-v1", render_mode="human")
```

produced:

```text
Total reward:   500.0
Episode length: 500
Terminated:     False
Truncated:      True
```

The episode therefore reached the 500-step time limit without terminating due
to failure.

---

## Results plot

The evaluation comparison is stored in:

```text
plots/evaluation_comparison.png
```

The plot compares the two learning rates using only periodic evaluation
episodes rather than training rewards.

It shows that both configurations can reach the maximum reward, while
`lr=0.005` provides much better final stability.

---

## Project structure

```text
DLA_LAB3/
├── models.py
├── reinforce.py
└── Exercise1/
    ├── main.py
    ├── plot_results.py
    ├── README.md
    ├── plots/
    │   └── evaluation_comparison.png
    └── runs/
        ├── lr0.01_seed42/
        │   ├── config.json
        │   ├── training_metrics.csv
        │   └── evaluation_metrics.csv
        │
        └── lr0.005_seed42_final/
            ├── config.json
            ├── training_metrics.csv
            ├── evaluation_metrics.csv
            └── policy.pt
```

`models.py` contains the policy network.

`reinforce.py` contains trajectory collection, discounted-return computation,
the REINFORCE update, training and evaluation.

`Exercise1/main.py` defines the experiment configuration and saves the
artifacts.

---

## Run the experiment

From:

```text
DLA_LAB3/
```

activate the DRL environment and run:

```bash
conda activate DRL
python -m Exercise1.main
```

Generate the comparison plot with:

```bash
python -m Exercise1.plot_results
```

---

## Main conclusion

The Exercise 1 implementation shows that vanilla REINFORCE is capable of
learning CartPole using a very small policy network.

With Adam and learning rate `0.01`, the agent reaches high rewards but can lose
the learned behaviour later in training.

Reducing the learning rate to `0.005` significantly improves the final
stability: the final six periodic evaluations all reach the maximum average
reward of 500.

The experiment also highlights the variability of vanilla REINFORCE, motivating
the variance-reduction techniques introduced in Exercise 2.
