import csv
import json
import random
from pathlib import Path

import gymnasium as gym
import torch

from Exercise3.dqn import (
    QNetwork,
    ReplayBuffer,
    train_dqn,
)


SEED = 42

ENV_NAME = "CartPole-v1"

NUM_EPISODES = 250
GAMMA = 0.99

LEARNING_RATE = 5e-4
HIDDEN_DIM = 64

LOSS_TYPE = "huber"

BUFFER_CAPACITY = 10_000
BATCH_SIZE = 64
MIN_BUFFER_SIZE = 500

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 10_000

TARGET_SYNC_EVERY = 250

EVAL_EVERY = 10
EVAL_EPISODES = 10

RUN_NAME = "cartpole_dqn_huber_lr0.0005_seed42"


def get_output_dir():
    output_dir = (
        Path(__file__).parent
        / "runs"
        / RUN_NAME
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir


def save_results(
    training_history,
    evaluation_history,
    online_network,
    output_dir,
):
    config = {
        "run_name": RUN_NAME,
        "environment": ENV_NAME,
        "seed": SEED,
        "num_episodes": NUM_EPISODES,
        "gamma": GAMMA,
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam",
        "hidden_dim": HIDDEN_DIM,
        "loss": LOSS_TYPE,
        "buffer_capacity": BUFFER_CAPACITY,
        "batch_size": BATCH_SIZE,
        "min_buffer_size": MIN_BUFFER_SIZE,
        "epsilon_start": EPSILON_START,
        "epsilon_end": EPSILON_END,
        "epsilon_decay_steps": EPSILON_DECAY_STEPS,
        "target_sync_every": TARGET_SYNC_EVERY,
        "eval_every": EVAL_EVERY,
        "eval_episodes": EVAL_EPISODES,
        "target_update": "hard",
        "checkpoint_selection":
            "best_evaluation_average_reward",
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
        writer = csv.writer(file, lineterminator="\n")

        writer.writerow(
            [
                "episode",
                "total_steps",
                "reward",
                "length",
                "mean_loss",
                "epsilon",
            ]
        )

        for row in training_history:
            writer.writerow(
                [
                    row["episode"],
                    row["total_steps"],
                    row["reward"],
                    row["length"],
                    row["mean_loss"],
                    row["epsilon"],
                ]
            )

    with open(
        output_dir / "evaluation_metrics.csv",
        "w",
        newline="",
    ) as file:
        writer = csv.writer(file, lineterminator="\n")

        writer.writerow(
            [
                "episode",
                "total_steps",
                "average_reward",
                "average_length",
            ]
        )

        for row in evaluation_history:
            writer.writerow(
                [
                    row["episode"],
                    row["total_steps"],
                    row["average_reward"],
                    row["average_length"],
                ]
            )

    torch.save(
        online_network.state_dict(),
        output_dir / "final_q_network.pt",
    )


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    train_env = gym.make(
        ENV_NAME
    )

    eval_env = gym.make(
        ENV_NAME
    )

    train_env.reset(
        seed=SEED
    )

    eval_env.reset(
        seed=SEED + 1
    )

    train_env.action_space.seed(
        SEED
    )

    eval_env.action_space.seed(
        SEED + 1
    )

    state_dim = (
        train_env.observation_space.shape[0]
    )

    action_dim = (
        train_env.action_space.n
    )

    print(
        "Environment:",
        ENV_NAME,
    )

    print(
        "State dimension:",
        state_dim,
    )

    print(
        "Action dimension:",
        action_dim,
    )

    print(
        "Loss:",
        LOSS_TYPE,
    )

    online_network = QNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=HIDDEN_DIM,
    )

    target_network = QNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=HIDDEN_DIM,
    )

    optimizer = torch.optim.Adam(
        online_network.parameters(),
        lr=LEARNING_RATE,
    )

    replay_buffer = ReplayBuffer(
        capacity=BUFFER_CAPACITY,
    )

    output_dir = get_output_dir()

    (
        training_history,
        evaluation_history,
        total_steps,
        updates,
    ) = train_dqn(
        env=train_env,
        eval_env=eval_env,
        online_network=online_network,
        target_network=target_network,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        num_episodes=NUM_EPISODES,
        gamma=GAMMA,
        batch_size=BATCH_SIZE,
        min_buffer_size=MIN_BUFFER_SIZE,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay_steps=EPSILON_DECAY_STEPS,
        target_sync_every=TARGET_SYNC_EVERY,
        eval_every=EVAL_EVERY,
        eval_episodes=EVAL_EPISODES,
        checkpoint_path=(
            output_dir
            / "best_q_network.pt"
        ),
        loss_type=LOSS_TYPE,
    )

    train_env.close()
    eval_env.close()

    save_results(
        training_history=training_history,
        evaluation_history=evaluation_history,
        online_network=online_network,
        output_dir=output_dir,
    )

    print("\nTraining completed")

    print(
        "Episodes:",
        len(training_history),
    )

    print(
        "Environment steps:",
        total_steps,
    )

    print(
        "Optimizer updates:",
        updates,
    )

    print(
        "Evaluations:",
        len(evaluation_history),
    )

    print(
        "Artifacts:",
        output_dir,
    )

    if evaluation_history:
        best = max(
            evaluation_history,
            key=lambda row:
                row["average_reward"],
        )

        final = evaluation_history[-1]

        print(
            "Best evaluation:",
            f"episode={best['episode']}",
            f"reward={best['average_reward']:.2f}",
        )

        print(
            "Final evaluation:",
            f"episode={final['episode']}",
            f"reward={final['average_reward']:.2f}",
        )


if __name__ == "__main__":
    main()
