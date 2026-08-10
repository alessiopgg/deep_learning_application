import csv
import json
from pathlib import Path

import gymnasium as gym
import torch

from models import PolicyNetwork
from Exercise2.reinforce_ex2 import train


SEED = 42

NUM_EPISODES = 1000
GAMMA = 0.99

LEARNING_RATE = 0.005
HIDDEN_DIM = 64

EVAL_EVERY = 25
EVAL_EPISODES = 20


RUNS = {
    "no_standardization_seed42": False,
    "standardized_returns_seed42": True,
}


def save_results(
    run_name,
    standardize,
    episode_rewards,
    losses,
    evaluation_history,
    policy,
):
    output_dir = Path(__file__).parent / "runs" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_name": run_name,
        "seed": SEED,
        "num_episodes": NUM_EPISODES,
        "gamma": GAMMA,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "eval_every": EVAL_EVERY,
        "eval_episodes": EVAL_EPISODES,
        "hidden_dim": HIDDEN_DIM,
        "activation": "ReLU",
        "standardize_returns": standardize,
    }

    with open(output_dir / "config.json", "w") as file:
        json.dump(config, file, indent=4)

    with open(
        output_dir / "training_metrics.csv",
        "w",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            ["episode", "reward", "policy_loss"]
        )

        for episode, (reward, loss) in enumerate(
            zip(episode_rewards, losses),
            start=1,
        ):
            writer.writerow(
                [episode, reward, loss]
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


def run_experiment(
    run_name,
    standardize,
):
    print("\n" + "=" * 70)
    print(f"Run: {run_name}")
    print(f"Standardize returns: {standardize}")
    print("=" * 70)

    torch.manual_seed(SEED)

    train_env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")

    train_env.reset(seed=SEED)
    eval_env.reset(seed=SEED + 1)

    policy = PolicyNetwork(
        hidden_dim=HIDDEN_DIM,
    )

    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=LEARNING_RATE,
    )

    episode_rewards, losses, evaluation_history = train(
        env=train_env,
        eval_env=eval_env,
        policy=policy,
        optimizer=optimizer,
        num_episodes=NUM_EPISODES,
        gamma=GAMMA,
        eval_every=EVAL_EVERY,
        eval_episodes=EVAL_EPISODES,
        standardize=standardize,
    )

    train_env.close()
    eval_env.close()

    save_results(
        run_name=run_name,
        standardize=standardize,
        episode_rewards=episode_rewards,
        losses=losses,
        evaluation_history=evaluation_history,
        policy=policy,
    )

    print("\nRun completed")
    print("Training episodes:", len(episode_rewards))
    print("Evaluations:", len(evaluation_history))


def main():
    for run_name, standardize in RUNS.items():
        run_experiment(
            run_name=run_name,
            standardize=standardize,
        )

    print("\nAll Exercise 2 standardization runs completed.")


if __name__ == "__main__":
    main()