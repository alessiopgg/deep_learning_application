import copy

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from reinforce import (
    compute_discounted_returns,
    evaluate_policy,
)


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

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action.item())

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

    policy_loss = -(
        log_probs_tensor * returns_tensor
    ).sum()

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    return policy_loss.item()


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

    # V_w(S_t)
    values = value_network(states_tensor)

    # Advantage:
    # A_t = G_t - V_w(S_t)
    #
    # detach() is essential here:
    # the policy loss must update only the policy,
    # not the ValueNetwork.
    advantages = returns_tensor - values.detach()

    policy_loss = -(
        log_probs_tensor * advantages
    ).sum()

    # The ValueNetwork learns to approximate
    # the Monte Carlo return G_t.
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
    policy_losses = []
    evaluation_history = []

    best_reward = float("-inf")
    best_episode = None
    best_policy_state = None

    for episode in range(1, num_episodes + 1):
        (
            _,
            log_probs,
            rewards,
            _,
            _,
        ) = collect_episode(
            env,
            policy,
        )

        policy_loss = update_policy(
            log_probs=log_probs,
            rewards=rewards,
            optimizer=optimizer,
            gamma=gamma,
            standardize=standardize,
        )

        total_reward = sum(rewards)

        episode_rewards.append(total_reward)
        policy_losses.append(policy_loss)

        if episode % eval_every == 0:
            (
                average_reward,
                average_length,
            ) = evaluate_policy(
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

            if average_reward > best_reward:
                best_reward = average_reward
                best_episode = episode
                best_policy_state = copy.deepcopy(
                    policy.state_dict()
                )

            print(
                f"Episode {episode}/{num_episodes} "
                f"- eval reward: {average_reward:.2f} "
                f"- eval length: {average_length:.2f}"
            )

    if best_policy_state is None:
        raise RuntimeError(
            "No evaluation was performed during training. "
            "Use eval_every <= num_episodes."
        )

    best_checkpoint = {
        "state_dict": best_policy_state,
        "episode": best_episode,
        "average_reward": best_reward,
    }

    return (
        episode_rewards,
        policy_losses,
        evaluation_history,
        best_checkpoint,
    )


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

    best_reward = float("-inf")
    best_episode = None
    best_policy_state = None
    best_value_state = None

    for episode in range(1, num_episodes + 1):
        (
            states,
            log_probs,
            rewards,
            _,
            _,
        ) = collect_episode(
            env,
            policy,
        )

        (
            policy_loss,
            value_loss,
        ) = update_policy_and_value(
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
            (
                average_reward,
                average_length,
            ) = evaluate_policy(
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

            # The best checkpoint is selected according
            # to policy performance, not value loss.
            if average_reward > best_reward:
                best_reward = average_reward
                best_episode = episode

                best_policy_state = copy.deepcopy(
                    policy.state_dict()
                )

                best_value_state = copy.deepcopy(
                    value_network.state_dict()
                )

            print(
                f"Episode {episode}/{num_episodes} "
                f"- eval reward: {average_reward:.2f} "
                f"- eval length: {average_length:.2f} "
                f"- value loss: {value_loss:.4f}"
            )

    if (
        best_policy_state is None
        or best_value_state is None
    ):
        raise RuntimeError(
            "No evaluation was performed during training. "
            "Use eval_every <= num_episodes."
        )

    best_checkpoint = {
        "policy_state_dict": best_policy_state,
        "value_state_dict": best_value_state,
        "episode": best_episode,
        "average_reward": best_reward,
    }

    return (
        episode_rewards,
        policy_losses,
        value_losses,
        evaluation_history,
        best_checkpoint,
    )