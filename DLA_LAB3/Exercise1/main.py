import csv
import json
from pathlib import Path

import gymnasium as gym
import torch

from models import PolicyNetwork
from reinforce import train


SEED = 42

NUM_EPISODES = 1000
GAMMA = 0.99

LEARNING_RATE = 0.005

HIDDEN_DIM = 64

EVAL_EVERY = 25
EVAL_EPISODES = 20

RUN_NAME = "lr0.005_seed42_final"

def save_results(
    episode_rewards,
    losses,
    evaluation_history,
    policy,
):
    output_dir = Path(__file__).parent / "runs" / RUN_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_name": RUN_NAME,
        "seed": SEED,
        "num_episodes": NUM_EPISODES,
        "gamma": GAMMA,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "eval_every": EVAL_EVERY,
        "eval_episodes": EVAL_EPISODES,
        "hidden_dim": HIDDEN_DIM,
        "activation": "ReLU",
    }

    with open(output_dir / "config.json", "w") as file:
        json.dump(config, file, indent=4)

    with open(output_dir / "training_metrics.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "reward", "loss"])

        for episode, (reward, loss) in enumerate(
            zip(episode_rewards, losses),
            start=1,
        ):
            writer.writerow([episode, reward, loss])

    with open(
        output_dir / "evaluation_metrics.csv",
        "w",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            ["episode", "average_reward", "average_length"]
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

def main():
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
    )

    train_env.close()
    eval_env.close()

    save_results(
    episode_rewards,
    losses,
    evaluation_history,
    policy,
)

    print("\nTraining completed")
    print("Training episodes:", len(episode_rewards))
    print("Evaluations:", len(evaluation_history))


if __name__ == "__main__":
    main()
