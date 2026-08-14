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


TRAINING_SEEDS = [
    42,
    123,
    456,
    789,
    1000,
]

EVAL_SEEDS = list(
    range(1000, 1100)
)


TRAINING_EPISODES = 2000

POLICY_LEARNING_RATE = 0.001
VALUE_LEARNING_RATE = 0.001

HIDDEN_DIM = 64


METHODS = [
    "vanilla",
    "standardized",
    "value_baseline",
]


def build_run_dir(
    method,
    seed,
):
    if method == "vanilla":
        run_name = (
            f"ex2_vanilla"
            f"_ep{TRAINING_EPISODES}"
            f"_lr{POLICY_LEARNING_RATE:g}"
            f"_seed{seed}"
        )

    elif method == "standardized":
        run_name = (
            f"ex2_standardized"
            f"_ep{TRAINING_EPISODES}"
            f"_lr{POLICY_LEARNING_RATE:g}"
            f"_seed{seed}"
        )

    elif method == "value_baseline":
        run_name = (
            f"ex2_value_baseline"
            f"_ep{TRAINING_EPISODES}"
            f"_plr{POLICY_LEARNING_RATE:g}"
            f"_vlr{VALUE_LEARNING_RATE:g}"
            f"_seed{seed}"
        )

    else:
        raise ValueError(
            f"Unknown method: {method}"
        )

    return RUNS_DIR / run_name


def load_policy(
    checkpoint_path,
):
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

            # Same environment initialization
            # for every checkpoint.
            observation, info = env.reset(
                seed=seed
            )

            # The policy is stochastic.
            # Resetting the PyTorch RNG ensures that
            # every checkpoint is evaluated using the
            # same action-sampling seed.
            torch.manual_seed(
                seed
            )

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

                action = (
                    distribution.sample()
                )

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
        "success_rate_500": np.mean(
            rewards >= 500.0
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

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "evaluation_seed",
                "reward",
                "episode_length",
            ]
        )

        for (
            seed,
            reward,
            length,
        ) in zip(
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
        "============================================================"
    )
    print(
        "EXERCISE 2 - ROBUST CHECKPOINT EVALUATION"
    )
    print(
        "============================================================"
    )

    print(
        f"Methods: {len(METHODS)}"
    )

    print(
        f"Training seeds: "
        f"{len(TRAINING_SEEDS)}"
    )

    print(
        f"Evaluation episodes per checkpoint: "
        f"{len(EVAL_SEEDS)}"
    )

    print(
        f"Evaluation seeds: "
        f"{EVAL_SEEDS[0]}-{EVAL_SEEDS[-1]}"
    )

    print(
        f"Total checkpoints: "
        f"{len(METHODS) * len(TRAINING_SEEDS) * 2}"
    )

    print()

    for method in METHODS:

        print(
            "############################################################"
        )

        print(
            f"METHOD: {method.upper()}"
        )

        print(
            "############################################################"
        )

        print()

        for training_seed in TRAINING_SEEDS:

            run_dir = build_run_dir(
                method,
                training_seed,
            )

            if not run_dir.exists():
                raise FileNotFoundError(
                    f"Run directory not found: "
                    f"{run_dir}"
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

            for (
                checkpoint_type,
                checkpoint_path,
            ) in checkpoints.items():

                if not checkpoint_path.exists():
                    raise FileNotFoundError(
                        f"Checkpoint not found: "
                        f"{checkpoint_path}"
                    )

                print(
                    "------------------------------------------------------------"
                )

                print(
                    f"Method: {method}"
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

                (
                    rewards,
                    lengths,
                ) = evaluate_checkpoint(
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
                        f"{method}"
                        f"_seed{training_seed}"
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
                        "method": method,
                        "training_seed": training_seed,
                        "checkpoint": checkpoint_type,
                        **summary,
                    }
                )

    # --------------------------------------------------------
    # Complete checkpoint summary
    # --------------------------------------------------------

    summary_path = (
        OUTPUT_DIR
        / "checkpoint_summary.csv"
    )

    fieldnames = [
        "method",
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

    with open(
        summary_path,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in summary_rows:
            writer.writerow(
                row
            )

    # --------------------------------------------------------
    # Aggregated comparison
    # --------------------------------------------------------

    print(
        "============================================================"
    )

    print(
        "AGGREGATED ROBUST EVALUATION"
    )

    print(
        "============================================================"
    )

    print()

    aggregated_rows = []

    for method in METHODS:

        for checkpoint_type in [
            "best",
            "final",
        ]:

            rows = [
                row
                for row in summary_rows
                if (
                    row["method"] == method
                    and
                    row["checkpoint"]
                    == checkpoint_type
                )
            ]

            mean_rewards = np.array(
                [
                    row["mean_reward"]
                    for row in rows
                ],
                dtype=np.float64,
            )

            success_rates = np.array(
                [
                    row["success_rate_500"]
                    for row in rows
                ],
                dtype=np.float64,
            )

            mean_reward = (
                mean_rewards.mean()
            )

            std_reward = (
                mean_rewards.std()
            )

            mean_success_rate = (
                success_rates.mean()
            )

            print(
                f"{method:15s} | "
                f"{checkpoint_type:5s} | "
                f"reward "
                f"{mean_reward:.2f} "
                f"± {std_reward:.2f} | "
                f"success @500 "
                f"{100 * mean_success_rate:.2f}%"
            )

            aggregated_rows.append(
                {
                    "method": method,
                    "checkpoint": checkpoint_type,
                    "mean_reward_across_training_seeds":
                        mean_reward,
                    "std_reward_across_training_seeds":
                        std_reward,
                    "mean_success_rate_500":
                        mean_success_rate,
                }
            )

    # --------------------------------------------------------
    # Aggregated CSV
    # --------------------------------------------------------

    aggregated_path = (
        OUTPUT_DIR
        / "aggregated_summary.csv"
    )

    with open(
        aggregated_path,
        "w",
        newline="",
    ) as file:

        fieldnames = [
            "method",
            "checkpoint",
            "mean_reward_across_training_seeds",
            "std_reward_across_training_seeds",
            "mean_success_rate_500",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in aggregated_rows:
            writer.writerow(
                row
            )

    print()

    print(
        "Saved checkpoint summary:"
    )

    print(
        summary_path
    )

    print()

    print(
        "Saved aggregated summary:"
    )

    print(
        aggregated_path
    )

    print()

    print(
        "Robust evaluation completed successfully."
    )


if __name__ == "__main__":
    main()