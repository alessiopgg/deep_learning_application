import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(128, 128)):
        super().__init__()

        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                ]
            )
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, state):
        return self.network(state)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        terminated,
        truncated,
    ):
        self.buffer.append(
            (
                torch.as_tensor(state, dtype=torch.float32).clone(),
                int(action),
                float(reward),
                torch.as_tensor(next_state, dtype=torch.float32).clone(),
                bool(terminated),
                bool(truncated),
            )
        )

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        (
            states,
            actions,
            rewards,
            next_states,
            terminated,
            truncated,
        ) = zip(*transitions)

        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(terminated, dtype=torch.bool),
            torch.tensor(truncated, dtype=torch.bool),
        )


def compute_dqn_loss(
    online_network,
    target_network,
    states,
    actions,
    rewards,
    next_states,
    terminated,
    gamma,
    loss_type,
):
    q_values = online_network(states)
    selected_q_values = q_values.gather(
        1,
        actions.unsqueeze(1),
    ).squeeze(1)

    with torch.no_grad():
        next_q_values = target_network(next_states).max(dim=1).values
        targets = (
            rewards
            + gamma
            * (~terminated).float()
            * next_q_values
        )

    if loss_type == "mse":
        return F.mse_loss(selected_q_values, targets)

    if loss_type == "huber":
        return F.smooth_l1_loss(selected_q_values, targets)

    raise ValueError(f"Unsupported loss type: {loss_type}")


def select_action(network, state, action_dim, epsilon):
    if random.random() < epsilon:
        return random.randrange(action_dim)

    state = torch.as_tensor(state, dtype=torch.float32)

    with torch.no_grad():
        return int(network(state).argmax().item())


def sync_target_network(online_network, target_network):
    target_network.load_state_dict(online_network.state_dict())


def soft_update_target_network(
    online_network,
    target_network,
    tau,
):
    with torch.no_grad():
        for target_parameter, online_parameter in zip(
            target_network.parameters(),
            online_network.parameters(),
        ):
            target_parameter.mul_(1.0 - tau)
            target_parameter.add_(online_parameter, alpha=tau)


def linear_epsilon(
    step,
    epsilon_start,
    epsilon_end,
    decay_steps,
):
    fraction = min(step / decay_steps, 1.0)
    return (
        epsilon_start
        + fraction * (epsilon_end - epsilon_start)
    )


def evaluate_dqn(env, network, seeds):
    rewards = []
    lengths = []

    was_training = network.training
    network.eval()

    with torch.inference_mode():
        for seed in seeds:
            state, _ = env.reset(seed=seed)
            terminated = False
            truncated = False
            total_reward = 0.0
            length = 0

            while not (terminated or truncated):
                state_tensor = torch.as_tensor(
                    state,
                    dtype=torch.float32,
                )
                action = int(network(state_tensor).argmax().item())
                (
                    state,
                    reward,
                    terminated,
                    truncated,
                    _,
                ) = env.step(action)

                total_reward += reward
                length += 1

            rewards.append(total_reward)
            lengths.append(length)

    if was_training:
        network.train()

    return (
        sum(rewards) / len(rewards),
        sum(lengths) / len(lengths),
    )


def train_dqn(
    env,
    eval_env,
    online_network,
    target_network,
    optimizer,
    replay_buffer,
    num_episodes,
    gamma,
    batch_size,
    min_buffer_size,
    epsilon_start,
    epsilon_end,
    epsilon_decay_steps,
    train_frequency,
    target_update_mode,
    target_sync_every,
    target_tau,
    eval_every,
    eval_seeds,
    checkpoint_path,
    loss_type,
    gradient_clip_norm,
    training_seed,
    early_stopping_reward=None,
    early_stopping_patience=0,
):
    training_history = []
    evaluation_history = []

    total_steps = 0
    updates = 0
    best_evaluation_reward = float("-inf")
    successful_evaluations = 0

    sync_target_network(online_network, target_network)

    for episode in range(1, num_episodes + 1):
        if episode == 1:
            state, _ = env.reset(seed=training_seed)
        else:
            state, _ = env.reset()

        terminated = False
        truncated = False
        episode_reward = 0.0
        episode_length = 0
        losses = []
        epsilon = epsilon_start

        while not (terminated or truncated):
            epsilon = linear_epsilon(
                total_steps,
                epsilon_start,
                epsilon_end,
                epsilon_decay_steps,
            )

            action = select_action(
                online_network,
                state,
                env.action_space.n,
                epsilon,
            )

            (
                next_state,
                reward,
                terminated,
                truncated,
                _,
            ) = env.step(action)

            replay_buffer.push(
                state,
                action,
                reward,
                next_state,
                terminated,
                truncated,
            )

            state = next_state
            episode_reward += reward
            episode_length += 1
            total_steps += 1

            if (
                len(replay_buffer) >= min_buffer_size
                and len(replay_buffer) >= batch_size
                and total_steps % train_frequency == 0
            ):
                (
                    states,
                    actions,
                    rewards,
                    next_states,
                    terminated_batch,
                    _,
                ) = replay_buffer.sample(batch_size)

                loss = compute_dqn_loss(
                    online_network,
                    target_network,
                    states,
                    actions,
                    rewards,
                    next_states,
                    terminated_batch,
                    gamma,
                    loss_type,
                )

                optimizer.zero_grad()
                loss.backward()

                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        online_network.parameters(),
                        gradient_clip_norm,
                    )

                optimizer.step()
                updates += 1
                losses.append(loss.item())

                if target_update_mode == "soft":
                    soft_update_target_network(
                        online_network,
                        target_network,
                        target_tau,
                    )
                elif updates % target_sync_every == 0:
                    sync_target_network(
                        online_network,
                        target_network,
                    )

        training_history.append(
            {
                "episode": episode,
                "total_steps": total_steps,
                "reward": episode_reward,
                "length": episode_length,
                "mean_loss": (
                    sum(losses) / len(losses)
                    if losses
                    else None
                ),
                "epsilon": epsilon,
            }
        )

        if episode % eval_every != 0:
            continue

        average_reward, average_length = evaluate_dqn(
            eval_env,
            online_network,
            eval_seeds,
        )

        evaluation_history.append(
            {
                "episode": episode,
                "total_steps": total_steps,
                "average_reward": average_reward,
                "average_length": average_length,
            }
        )

        if average_reward > best_evaluation_reward:
            best_evaluation_reward = average_reward
            torch.save(
                online_network.state_dict(),
                checkpoint_path,
            )

        print(
            f"Episode {episode}/{num_episodes} "
            f"- steps: {total_steps} "
            f"- epsilon: {epsilon:.3f} "
            f"- evaluation reward: {average_reward:.2f}"
        )

        if (
            early_stopping_reward is not None
            and early_stopping_patience > 0
        ):
            if average_reward >= early_stopping_reward:
                successful_evaluations += 1
            else:
                successful_evaluations = 0

            if successful_evaluations >= early_stopping_patience:
                print("Early stopping criterion reached.")
                break

    return (
        training_history,
        evaluation_history,
        total_steps,
        updates,
    )
