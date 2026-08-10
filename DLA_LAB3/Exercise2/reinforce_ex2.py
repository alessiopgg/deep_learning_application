import torch
import torch.nn.functional as F

from reinforce import (
    compute_discounted_returns,
    evaluate_policy,
)

from torch.distributions import Categorical


def collect_episode(env, policy):
    observation, info = env.reset()

    states = []
    log_probs = []
    rewards = []

    terminated = False
    truncated = False

    while not (terminated or truncated):
        state = torch.tensor(
            observation,
            dtype=torch.float32,
        )

        logits = policy(state)
        distribution = Categorical(logits=logits)

        action = distribution.sample()
        log_prob = distribution.log_prob(action)

        observation, reward, terminated, truncated, info = env.step(
            action.item()
        )

        states.append(state)
        log_probs.append(log_prob)
        rewards.append(reward)

    return (
        states,
        log_probs,
        rewards,
        terminated,
        truncated,
    )

def standardize_returns(
    returns: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    mean = returns.mean()
    std = returns.std(correction=0)

    return (returns - mean) / (std + eps)


def update_policy(
    log_probs,
    rewards,
    optimizer,
    gamma,
    standardize,
):
    log_probs_tensor = torch.stack(log_probs)

    returns_tensor = compute_discounted_returns(
        rewards,
        gamma,
    )

    if standardize:
        returns_tensor = standardize_returns(
            returns_tensor,
        )

    loss = -(
        log_probs_tensor * returns_tensor
    ).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def update_policy_and_value(
    states,
    log_probs,
    rewards,
    policy_optimizer,
    value_network,
    value_optimizer,
    gamma,
):
    states_tensor = torch.stack(states)
    log_probs_tensor = torch.stack(log_probs)

    returns_tensor = compute_discounted_returns(
        rewards,
        gamma,
    )

    values = value_network(states_tensor)

    advantages = returns_tensor - values.detach()

    policy_loss = -(
        log_probs_tensor * advantages
    ).sum()

    value_loss = F.mse_loss(
        values,
        returns_tensor,
    )

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    return (
        policy_loss.item(),
        value_loss.item(),
    )

def train(
    env,
    eval_env,
    policy,
    optimizer,
    num_episodes,
    gamma,
    eval_every,
    eval_episodes,
    standardize,
):
    episode_rewards = []
    losses = []
    evaluation_history = []

    for episode in range(1, num_episodes + 1):
        states, log_probs, rewards, _, _ = collect_episode(
            env,
            policy,
        )

        loss = update_policy(
            log_probs=log_probs,
            rewards=rewards,
            optimizer=optimizer,
            gamma=gamma,
            standardize=standardize,
        )

        total_reward = sum(rewards)

        episode_rewards.append(total_reward)
        losses.append(loss)

        if episode % eval_every == 0:
            average_reward, average_length = evaluate_policy(
                env=eval_env,
                policy=policy,
                num_episodes=eval_episodes,
            )

            evaluation_history.append(
                {
                    "episode": episode,
                    "average_reward": average_reward,
                    "average_length": average_length,
                }
            )

            print(
                f"Episode {episode}/{num_episodes} "
                f"- eval reward: {average_reward:.2f} "
                f"- eval length: {average_length:.2f}"
            )

    return episode_rewards, losses, evaluation_history

def train_with_value_baseline(
    env,
    eval_env,
    policy,
    value_network,
    policy_optimizer,
    value_optimizer,
    num_episodes,
    gamma,
    eval_every,
    eval_episodes,
):
    episode_rewards = []
    policy_losses = []
    value_losses = []
    evaluation_history = []

    for episode in range(1, num_episodes + 1):
        states, log_probs, rewards, _, _ = collect_episode(
            env,
            policy,
        )

        policy_loss, value_loss = update_policy_and_value(
            states=states,
            log_probs=log_probs,
            rewards=rewards,
            policy_optimizer=policy_optimizer,
            value_network=value_network,
            value_optimizer=value_optimizer,
            gamma=gamma,
        )

        total_reward = sum(rewards)

        episode_rewards.append(total_reward)
        policy_losses.append(policy_loss)
        value_losses.append(value_loss)

        if episode % eval_every == 0:
            average_reward, average_length = evaluate_policy(
                env=eval_env,
                policy=policy,
                num_episodes=eval_episodes,
            )

            evaluation_history.append(
                {
                    "episode": episode,
                    "average_reward": average_reward,
                    "average_length": average_length,
                }
            )

            print(
                f"Episode {episode}/{num_episodes} "
                f"- eval reward: {average_reward:.2f} "
                f"- eval length: {average_length:.2f} "
                f"- value loss: {value_loss:.4f}"
            )

    return (
        episode_rewards,
        policy_losses,
        value_losses,
        evaluation_history,
    )