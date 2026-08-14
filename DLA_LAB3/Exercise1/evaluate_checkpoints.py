import csv
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Categorical

from models import PolicyNetwork


BASE_DIR = Path(__file__).parent
RUNS_DIR = BASE_DIR / "runs"

OUTPUT_DIR = BASE_DIR / "robust_evaluation"

SEEDS = [42, 123, 456, 789, 1000]

EVAL_SEEDS = list(range(1000, 1100))

LEARNING_RATE = 0.001
GAMMA = 0.99
HIDDEN_DIM = 64
TRAINING_EPISODES = 2000


def build_run_dir(seed):
    run_name = (
        f"reinforce_extended{TRAINING_EPISODES}"
        f"_lr{LEARNING_RATE:g}"
        f"_gamma{GAMMA:g}"
        f"_h{HIDDEN_DIM}"
        f"_seed{seed}"
    )

    return RUNS_DIR / run_name


def load_policy(checkpoint_path):
    policy = PolicyNetwork(
        hidden_dim=HIDDEN_DIM,
    )

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    policy.load_state_dict(
        state_dict
    )

    policy.eval()

    return policy


def evaluate_checkpoint(
    policy,
    evaluation_seeds,
):
    rewards = []
    lengths = []

    env = gym.make(
        "CartPole-v1"
    )

    with torch.inference_mode():
        for seed in evaluation_seeds:

            # Same environment seed for every checkpoint.
            observation, info = env.reset(
                seed=seed
            )

            # Also reset PyTorch RNG so stochastic action sampling
            # starts from the same RNG seed for each evaluation episode.
            torch.manual_seed(seed)

            terminated = False
            truncated = False

            total_reward = 0.0
            episode_length = 0

            while not (
                terminated
                or truncated
            ):
                state = torch.tensor(
                    observation,
                    dtype=torch.float32,
                )

                logits = policy(
                    state
                )

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

            rewards.append(
                total_reward
            )

            lengths.append(
                episode_length
            )

    env.close()

    return (
        np.array(
            rewards,
            dtype=np.float64,
        ),
        np.array(
            lengths,
            dtype=np.float64,
        ),
    )


def compute_summary(
    rewards,
    lengths,
):
    return {
        "mean_reward": rewards.mean(),
        "std_reward": rewards.std(),
        "median_reward": np.median(
            rewards
        ),
        "min_reward": rewards.min(),
        "max_reward": rewards.max(),
        "mean_length": lengths.mean(),
        "success_rate_500": (
            np.mean(rewards >= 500.0)
        ),
    }


def save_episode_metrics(
    output_path,
    evaluation_seeds,
    rewards,
    lengths,
):
    with open(
        output_path,
        "w",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "evaluation_seed",
                "reward",
                "episode_length",
            ]
        )

        for seed, reward, length in zip(
            evaluation_seeds,
            rewards,
            lengths,
        ):
            writer.writerow(
                [
                    seed,
                    reward,
                    int(length),
                ]
            )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows = []

    print(
        "============================================"
    )
    print(
        "ROBUST CHECKPOINT EVALUATION"
    )
    print(
        "============================================"
    )

    print(
        f"Evaluation episodes: "
        f"{len(EVAL_SEEDS)}"
    )

    print(
        f"Evaluation seeds: "
        f"{EVAL_SEEDS[0]}-{EVAL_SEEDS[-1]}"
    )

    print()

    for training_seed in SEEDS:
        run_dir = build_run_dir(
            training_seed
        )

        checkpoints = {
            "best": (
                run_dir
                / "best_policy.pt"
            ),
            "final": (
                run_dir
                / "policy.pt"
            ),
        }

        for checkpoint_type, checkpoint_path in checkpoints.items():

            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint not found: "
                    f"{checkpoint_path}"
                )

            print(
                "--------------------------------------------"
            )

            print(
                f"Training seed: "
                f"{training_seed}"
            )

            print(
                f"Checkpoint: "
                f"{checkpoint_type}"
            )

            policy = load_policy(
                checkpoint_path
            )

            rewards, lengths = evaluate_checkpoint(
                policy,
                EVAL_SEEDS,
            )

            summary = compute_summary(
                rewards,
                lengths,
            )

            print(
                f"Mean reward: "
                f"{summary['mean_reward']:.2f}"
            )

            print(
                f"Std reward: "
                f"{summary['std_reward']:.2f}"
            )

            print(
                f"Median reward: "
                f"{summary['median_reward']:.2f}"
            )

            print(
                f"Min reward: "
                f"{summary['min_reward']:.2f}"
            )

            print(
                f"Max reward: "
                f"{summary['max_reward']:.2f}"
            )

            print(
                f"Success rate @500: "
                f"{100 * summary['success_rate_500']:.1f}%"
            )

            print()

            output_path = (
                OUTPUT_DIR
                / (
                    f"seed{training_seed}"
                    f"_{checkpoint_type}"
                    f"_episodes.csv"
                )
            )

            save_episode_metrics(
                output_path,
                EVAL_SEEDS,
                rewards,
                lengths,
            )

            summary_rows.append(
                {
                    "training_seed": training_seed,
                    "checkpoint": checkpoint_type,
                    **summary,
                }
            )

    summary_path = (
        OUTPUT_DIR
        / "checkpoint_summary.csv"
    )

    with open(
        summary_path,
        "w",
        newline="",
    ) as file:
        fieldnames = [
            "training_seed",
            "checkpoint",
            "mean_reward",
            "std_reward",
            "median_reward",
            "min_reward",
            "max_reward",
            "mean_length",
            "success_rate_500",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in summary_rows:
            writer.writerow(
                row
            )

    print(
        "============================================"
    )
    print(
        "AGGREGATED COMPARISON"
    )
    print(
        "============================================"
    )

    for checkpoint_type in [
        "best",
        "final",
    ]:
        rows = [
            row
            for row in summary_rows
            if row["checkpoint"]
            == checkpoint_type
        ]

        mean_rewards = np.array(
            [
                row["mean_reward"]
                for row in rows
            ]
        )

        success_rates = np.array(
            [
                row["success_rate_500"]
                for row in rows
            ]
        )

        print()
        print(
            f"{checkpoint_type.upper()} checkpoints"
        )

        print(
            "Mean reward across training seeds:",
            f"{mean_rewards.mean():.2f} "
            f"± {mean_rewards.std():.2f}",
        )

        print(
            "Mean success rate @500:",
            f"{100 * success_rates.mean():.2f}%",
        )

    print()
    print(
        "Saved summary:",
        summary_path,
    )

    print(
        "Robust evaluation completed successfully."
    )


if __name__ == "__main__":
    main()
    