import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class ReplayBuffer:
    def __init__(self, capacity: int):
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
        state = torch.as_tensor(
            state,
            dtype=torch.float32,
        ).detach().clone()

        next_state = torch.as_tensor(
            next_state,
            dtype=torch.float32,
        ).detach().clone()

        self.buffer.append(
            (
                state,
                int(action),
                float(reward),
                next_state,
                bool(terminated),
                bool(truncated),
            )
        )

    def sample(self, batch_size: int):
        if batch_size > len(self.buffer):
            raise ValueError(
                "batch_size cannot be larger than replay buffer size"
            )

        transitions = random.sample(
            self.buffer,
            batch_size,
        )

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
            torch.tensor(
                actions,
                dtype=torch.int64,
            ),
            torch.tensor(
                rewards,
                dtype=torch.float32,
            ),
            torch.stack(next_states),
            torch.tensor(
                terminated,
                dtype=torch.bool,
            ),
            torch.tensor(
                truncated,
                dtype=torch.bool,
            ),
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
    loss_type="mse",
):
    q_values = online_network(states)

    selected_q_values = q_values.gather(
        dim=1,
        index=actions.unsqueeze(1),
    ).squeeze(1)

    with torch.no_grad():
        next_q_values = target_network(
            next_states
        )

        max_next_q_values = next_q_values.max(
            dim=1
        ).values

        targets = (
            rewards
            + gamma
            * (~terminated).float()
            * max_next_q_values
        )

    if loss_type == "mse":
        loss = F.mse_loss(
            selected_q_values,
            targets,
        )

    elif loss_type == "huber":
        loss = F.smooth_l1_loss(
            selected_q_values,
            targets,
        )

    else:
        raise ValueError(
            f"Unsupported loss_type: {loss_type}"
        )

    return (
        loss,
        selected_q_values,
        targets,
    )


def select_action(
    network,
    state,
    action_dim,
    epsilon,
):
    if random.random() < epsilon:
        return random.randrange(action_dim)

    state_tensor = torch.as_tensor(
        state,
        dtype=torch.float32,
    )

    with torch.no_grad():
        q_values = network(
            state_tensor
        )

    return int(
        torch.argmax(q_values).item()
    )


def sync_target_network(
    online_network,
    target_network,
):
    target_network.load_state_dict(
        online_network.state_dict()
    )


def linear_epsilon(
    step,
    epsilon_start,
    epsilon_end,
    decay_steps,
):
    if decay_steps <= 0:
        raise ValueError(
            "decay_steps must be positive"
        )

    fraction = min(
        step / decay_steps,
        1.0,
    )

    return (
        epsilon_start
        + fraction
        * (epsilon_end - epsilon_start)
    )


def evaluate_dqn(
    env,
    network,
    num_episodes,
):
    total_rewards = []
    episode_lengths = []

    was_training = network.training
    network.eval()

    with torch.inference_mode():
        for _ in range(num_episodes):
            state, info = env.reset()

            terminated = False
            truncated = False

            total_reward = 0.0
            episode_length = 0

            while not (
                terminated or truncated
            ):
                state_tensor = torch.as_tensor(
                    state,
                    dtype=torch.float32,
                )

                q_values = network(
                    state_tensor
                )

                action = int(
                    torch.argmax(
                        q_values
                    ).item()
                )

                (
                    state,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action)

                total_reward += reward
                episode_length += 1

            total_rewards.append(
                total_reward
            )

            episode_lengths.append(
                episode_length
            )

    if was_training:
        network.train()

    average_reward = (
        sum(total_rewards)
        / num_episodes
    )

    average_length = (
        sum(episode_lengths)
        / num_episodes
    )

    return (
        average_reward,
        average_length,
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
    target_sync_every,
    eval_every,
    eval_episodes,
    checkpoint_path=None,
    loss_type="mse",
):
    training_history = []
    evaluation_history = []

    total_steps = 0
    updates = 0

    best_evaluation_reward = float(
        "-inf"
    )

    sync_target_network(
        online_network=online_network,
        target_network=target_network,
    )

    for episode in range(
        1,
        num_episodes + 1,
    ):
        state, info = env.reset()

        terminated = False
        truncated = False

        episode_reward = 0.0
        episode_length = 0
        episode_losses = []

        while not (
            terminated or truncated
        ):
            epsilon = linear_epsilon(
                step=total_steps,
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
                decay_steps=epsilon_decay_steps,
            )

            action = select_action(
                network=online_network,
                state=state,
                action_dim=env.action_space.n,
                epsilon=epsilon,
            )

            (
                next_state,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            replay_buffer.push(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                terminated=terminated,
                truncated=truncated,
            )

            state = next_state

            episode_reward += reward
            episode_length += 1
            total_steps += 1

            if (
                len(replay_buffer)
                >= min_buffer_size
                and len(replay_buffer)
                >= batch_size
            ):
                (
                    states,
                    actions,
                    rewards,
                    next_states,
                    terminated_batch,
                    truncated_batch,
                ) = replay_buffer.sample(
                    batch_size=batch_size,
                )

                (
                    loss,
                    selected_q_values,
                    targets,
                ) = compute_dqn_loss(
                    online_network=online_network,
                    target_network=target_network,
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated_batch,
                    gamma=gamma,
                    loss_type=loss_type,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                updates += 1

                episode_losses.append(
                    loss.item()
                )

                if (
                    updates
                    % target_sync_every
                    == 0
                ):
                    sync_target_network(
                        online_network=online_network,
                        target_network=target_network,
                    )

        mean_loss = (
            sum(episode_losses)
            / len(episode_losses)
            if episode_losses
            else None
        )

        training_history.append(
            {
                "episode": episode,
                "total_steps": total_steps,
                "reward": episode_reward,
                "length": episode_length,
                "mean_loss": mean_loss,
                "epsilon": epsilon,
            }
        )

        if episode % eval_every == 0:
            (
                average_reward,
                average_length,
            ) = evaluate_dqn(
                env=eval_env,
                network=online_network,
                num_episodes=eval_episodes,
            )

            evaluation_history.append(
                {
                    "episode": episode,
                    "total_steps": total_steps,
                    "average_reward": average_reward,
                    "average_length": average_length,
                }
            )

            if (
                average_reward
                > best_evaluation_reward
            ):
                best_evaluation_reward = (
                    average_reward
                )

                if checkpoint_path is not None:
                    torch.save(
                        online_network.state_dict(),
                        checkpoint_path,
                    )

            print(
                f"Episode {episode}/{num_episodes} "
                f"- steps: {total_steps} "
                f"- epsilon: {epsilon:.3f} "
                f"- eval reward: "
                f"{average_reward:.2f} "
                f"- eval length: "
                f"{average_length:.2f}"
            )

    return (
        training_history,
        evaluation_history,
        total_steps,
        updates,
    )
