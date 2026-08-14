import copy

import torch
from torch.distributions import Categorical


def collect_episode(env, policy):
    observation, info = env.reset()

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

        log_probs.append(log_prob)
        rewards.append(reward)

    return log_probs, rewards, terminated, truncated


def compute_discounted_returns(rewards, gamma):
    returns = []
    G = 0.0

    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)

    return torch.tensor(
        returns,
        dtype=torch.float32,
    )


def update_policy(
    log_probs,
    rewards,
    optimizer,
    gamma,
):
    log_probs_tensor = torch.stack(log_probs)

    returns_tensor = compute_discounted_returns(
        rewards,
        gamma,
    )

    loss = -(
        log_probs_tensor * returns_tensor
    ).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train(
    env,
    eval_env,
    policy,
    optimizer,
    num_episodes,
    gamma,
    eval_every,
    eval_episodes,
):
    episode_rewards = []
    losses = []
    evaluation_history = []

    best_reward = float("-inf")
    best_episode = None
    best_policy_state = None

    for episode in range(1, num_episodes + 1):
        log_probs, rewards, _, _ = collect_episode(
            env,
            policy,
        )

        loss = update_policy(
            log_probs,
            rewards,
            optimizer,
            gamma,
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
        losses,
        evaluation_history,
        best_checkpoint,
    )


def evaluate_policy(
    env,
    policy,
    num_episodes,
):
    total_rewards = []
    episode_lengths = []

    was_training = policy.training
    rng_state = torch.random.get_rng_state()

    policy.eval()

    with torch.inference_mode():
        for _ in range(num_episodes):
            observation, info = env.reset()

            terminated = False
            truncated = False

            total_reward = 0.0
            episode_length = 0

            while not (terminated or truncated):
                state = torch.tensor(
                    observation,
                    dtype=torch.float32,
                )

                logits = policy(state)

                distribution = Categorical(
                    logits=logits
                )

                action = distribution.sample()

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(
                    action.item()
                )

                total_reward += reward
                episode_length += 1

            total_rewards.append(total_reward)
            episode_lengths.append(episode_length)

    torch.random.set_rng_state(rng_state)

    if was_training:
        policy.train()

    average_reward = (
        sum(total_rewards) / num_episodes
    )

    average_length = (
        sum(episode_lengths) / num_episodes
    )

    return average_reward, average_length