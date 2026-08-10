import csv
import json
from pathlib import Path

import gymnasium as gym
import torch

from models import PolicyNetwork, ValueNetwork
from Exercise2.reinforce_ex2 import train_with_value_baseline


SEED = 42

NUM_EPISODES = 1000
GAMMA = 0.99

POLICY_LEARNING_RATE = 0.005
VALUE_LEARNING_RATE = 0.005

HIDDEN_DIM = 64

EVAL_EVERY = 25
EVAL_EPISODES = 20

RUN_NAME = "value_baseline_seed42"


def save_results(
    episode_rewards,
    policy_losses,
    value_losses,
    evaluation_history,
    policy,
    value_network,
):
    output_dir = (
        Path(__file__).parent
        / "runs"
        / RUN_NAME
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = {
        "run_name": RUN_NAME,
        "seed": SEED,
        "num_episodes": NUM_EPISODES,
        "gamma": GAMMA,

        "policy_architecture": "4-64-2",
        "policy_activation": "ReLU",
        "policy_optimizer": "Adam",
        "policy_learning_rate": POLICY_LEARNING_RATE,

        "value_architecture": "4-64-1",
        "value_activation": "ReLU",
        "value_optimizer": "Adam",
        "value_learning_rate": VALUE_LEARNING_RATE,

        "hidden_dim": HIDDEN_DIM,

        "eval_every": EVAL_EVERY,
        "eval_episodes": EVAL_EPISODES,

        "standardize_returns": False,
        "value_baseline": True,
    }

    with open(
        output_dir / "config.json",
        "w",
    ) as file:
        json.dump(
            config,
            file,
            indent=4,
        )

    with open(
        output_dir / "training_metrics.csv",
        "w",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "episode",
                "reward",
                "policy_loss",
                "value_loss",
            ]
        )

        for episode, (
            reward,
            policy_loss,
            value_loss,
        ) in enumerate(
            zip(
                episode_rewards,
                policy_losses,
                value_losses,
            ),
            start=1,
        ):
            writer.writerow(
                [
                    episode,
                    reward,
                    policy_loss,
                    value_loss,
                ]
            )

    with open(
        output_dir / "evaluation_metrics.csv",
        "w",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "episode",
                "average_reward",
                "average_length",
            ]
        )

        for evaluation in evaluation_history:
            writer.writerow(
                [
                    evaluation["episode"],
                    evaluation["average_reward"],
                    evaluation["average_length"],
                ]
            )

    torch.save(
        policy.state_dict(),
        output_dir / "policy.pt",
    )

    torch.save(
        value_network.state_dict(),
        output_dir / "value.pt",
    )


def main():
    torch.manual_seed(SEED)

    train_env = gym.make(
        "CartPole-v1"
    )

    eval_env = gym.make(
        "CartPole-v1"
    )

    train_env.reset(
        seed=SEED,
    )

    eval_env.reset(
        seed=SEED + 1,
    )

    policy = PolicyNetwork(
        hidden_dim=HIDDEN_DIM,
    )

    # Preserve the RNG state that the previous
    # experiments had immediately after policy
    # initialization.
    training_rng_state = (
        torch.random.get_rng_state()
    )

    value_network = ValueNetwork(
        hidden_dim=HIDDEN_DIM,
    )

    torch.random.set_rng_state(
        training_rng_state,
    )

    policy_optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=POLICY_LEARNING_RATE,
    )

    value_optimizer = torch.optim.Adam(
        value_network.parameters(),
        lr=VALUE_LEARNING_RATE,
    )

    (
        episode_rewards,
        policy_losses,
        value_losses,
        evaluation_history,
    ) = train_with_value_baseline(
        env=train_env,
        eval_env=eval_env,
        policy=policy,
        value_network=value_network,
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,
        num_episodes=NUM_EPISODES,
        gamma=GAMMA,
        eval_every=EVAL_EVERY,
        eval_episodes=EVAL_EPISODES,
    )

    train_env.close()
    eval_env.close()

    save_results(
        episode_rewards=episode_rewards,
        policy_losses=policy_losses,
        value_losses=value_losses,
        evaluation_history=evaluation_history,
        policy=policy,
        value_network=value_network,
    )

    print("\nValue-baseline training completed")
    print(
        "Training episodes:",
        len(episode_rewards),
    )
    print(
        "Evaluations:",
        len(evaluation_history),
    )


if __name__ == "__main__":
    main()