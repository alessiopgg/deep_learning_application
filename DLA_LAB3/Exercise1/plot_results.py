import csv
from pathlib import Path

import matplotlib

# Non-interactive backend: avoids Wayland / Qt display warnings on WSL/server.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).parent
RUNS_DIR = BASE_DIR / "runs"
PLOTS_DIR = BASE_DIR / "plots"

# ---------------------------------------------------------------------
# Experiment 1:
# comparison of three learning rates over five random seeds
# ---------------------------------------------------------------------

LEARNING_RATES = [0.001, 0.005, 0.01]
SEEDS = [42, 123, 456, 789, 1000]

GAMMA = 0.99
HIDDEN_DIM = 64

# ---------------------------------------------------------------------
# Experiment 2:
# extended training for the best/stablest learning rate
# ---------------------------------------------------------------------

EXTENDED_LEARNING_RATE = 0.001
EXTENDED_EPISODES = 2000
EXTENDED_SEEDS = [42, 123, 456, 789, 1000]

MOVING_AVERAGE_WINDOW = 50


# =====================================================================
# Utility functions
# =====================================================================


def build_run_dir(learning_rate, seed):
    run_name = (
        f"reinforce"
        f"_lr{learning_rate:g}"
        f"_gamma{GAMMA:g}"
        f"_h{HIDDEN_DIM}"
        f"_seed{seed}"
    )

    return RUNS_DIR / run_name


def build_extended_run_dir(seed):
    run_name = (
        f"reinforce_extended{EXTENDED_EPISODES}"
        f"_lr{EXTENDED_LEARNING_RATE:g}"
        f"_gamma{GAMMA:g}"
        f"_h{HIDDEN_DIM}"
        f"_seed{seed}"
    )

    return RUNS_DIR / run_name


def load_evaluation_metrics(run_dir):
    episodes = []
    average_rewards = []
    average_lengths = []

    csv_path = run_dir / "evaluation_metrics.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Evaluation metrics not found: {csv_path}"
        )

    with open(csv_path, newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            episodes.append(int(row["episode"]))
            average_rewards.append(
                float(row["average_reward"])
            )
            average_lengths.append(
                float(row["average_length"])
            )

    return (
        np.array(episodes),
        np.array(average_rewards),
        np.array(average_lengths),
    )


def load_training_metrics(run_dir):
    episodes = []
    rewards = []
    losses = []

    csv_path = run_dir / "training_metrics.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Training metrics not found: {csv_path}"
        )

    with open(csv_path, newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            episodes.append(
                int(row["episode"])
            )
            rewards.append(
                float(row["reward"])
            )
            losses.append(
                float(row["loss"])
            )

    return (
        np.array(episodes),
        np.array(rewards),
        np.array(losses),
    )


def moving_average(values, window):
    if window <= 1:
        return values

    kernel = np.ones(window) / window

    return np.convolve(
        values,
        kernel,
        mode="valid",
    )


# =====================================================================
# Load standard 1000-episode experiments
# =====================================================================


def load_all_runs():
    data = {}

    for learning_rate in LEARNING_RATES:
        data[learning_rate] = []

        for seed in SEEDS:
            run_dir = build_run_dir(
                learning_rate,
                seed,
            )

            print(
                f"Loading standard run: "
                f"lr={learning_rate:g}, "
                f"seed={seed}"
            )

            (
                eval_episodes,
                eval_rewards,
                eval_lengths,
            ) = load_evaluation_metrics(
                run_dir
            )

            (
                train_episodes,
                train_rewards,
                train_losses,
            ) = load_training_metrics(
                run_dir
            )

            data[learning_rate].append(
                {
                    "seed": seed,
                    "eval_episodes": eval_episodes,
                    "eval_rewards": eval_rewards,
                    "eval_lengths": eval_lengths,
                    "train_episodes": train_episodes,
                    "train_rewards": train_rewards,
                    "train_losses": train_losses,
                }
            )

    return data


# =====================================================================
# Load extended 2000-episode experiments
# =====================================================================


def load_extended_runs():
    runs = []

    for seed in EXTENDED_SEEDS:
        run_dir = build_extended_run_dir(
            seed
        )

        print(
            f"Loading extended run: "
            f"lr={EXTENDED_LEARNING_RATE:g}, "
            f"seed={seed}"
        )

        (
            eval_episodes,
            eval_rewards,
            eval_lengths,
        ) = load_evaluation_metrics(
            run_dir
        )

        (
            train_episodes,
            train_rewards,
            train_losses,
        ) = load_training_metrics(
            run_dir
        )

        runs.append(
            {
                "seed": seed,
                "eval_episodes": eval_episodes,
                "eval_rewards": eval_rewards,
                "eval_lengths": eval_lengths,
                "train_episodes": train_episodes,
                "train_rewards": train_rewards,
                "train_losses": train_losses,
            }
        )

    return runs


# =====================================================================
# Standard experiment plots
# =====================================================================


def plot_evaluation_mean_std(data):
    plt.figure(figsize=(11, 7))

    for learning_rate in LEARNING_RATES:
        runs = data[learning_rate]

        episodes = runs[0]["eval_episodes"]

        rewards = np.stack(
            [
                run["eval_rewards"]
                for run in runs
            ]
        )

        mean_rewards = rewards.mean(axis=0)
        std_rewards = rewards.std(axis=0)

        # Clip only the visualization band because CartPole reward
        # is physically bounded between 0 and 500.
        lower = np.clip(
            mean_rewards - std_rewards,
            0,
            500,
        )

        upper = np.clip(
            mean_rewards + std_rewards,
            0,
            500,
        )

        plt.plot(
            episodes,
            mean_rewards,
            linewidth=2,
            label=f"lr={learning_rate:g}",
        )

        plt.fill_between(
            episodes,
            lower,
            upper,
            alpha=0.2,
        )

    plt.axhline(
        y=500,
        linestyle="--",
        linewidth=1,
        label="Maximum reward (500)",
    )

    plt.xlabel("Training episode")
    plt.ylabel("Average evaluation reward")

    plt.title(
        "REINFORCE on CartPole-v1\n"
        "Evaluation reward across 5 random seeds"
    )

    plt.ylim(0, 520)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = (
        PLOTS_DIR
        / "evaluation_mean_std.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print("Saved:", output_path)


def plot_individual_evaluation_runs(data):
    for learning_rate in LEARNING_RATES:
        plt.figure(figsize=(11, 7))

        for run in data[learning_rate]:
            plt.plot(
                run["eval_episodes"],
                run["eval_rewards"],
                linewidth=1.5,
                label=f"seed={run['seed']}",
            )

        plt.axhline(
            y=500,
            linestyle="--",
            linewidth=1,
            label="Maximum reward (500)",
        )

        plt.xlabel("Training episode")
        plt.ylabel("Average evaluation reward")

        plt.title(
            "REINFORCE on CartPole-v1\n"
            f"Individual runs - lr={learning_rate:g}"
        )

        plt.ylim(0, 520)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        output_path = (
            PLOTS_DIR
            / (
                "evaluation_individual_"
                f"lr{learning_rate:g}.png"
            )
        )

        plt.savefig(
            output_path,
            dpi=200,
        )

        plt.close()

        print("Saved:", output_path)


def plot_final_reward_summary(data):
    means = []
    stds = []

    print()
    print("=== Final reward summary ===")

    for learning_rate in LEARNING_RATES:
        final_rewards = np.array(
            [
                run["eval_rewards"][-1]
                for run in data[learning_rate]
            ]
        )

        mean_reward = final_rewards.mean()
        std_reward = final_rewards.std()

        means.append(mean_reward)
        stds.append(std_reward)

        print()
        print(
            f"Learning rate: "
            f"{learning_rate:g}"
        )

        print(
            "Final rewards:",
            [
                round(float(value), 2)
                for value in final_rewards
            ],
        )

        print(
            f"Mean final reward: "
            f"{mean_reward:.2f}"
        )

        print(
            f"Std final reward: "
            f"{std_reward:.2f}"
        )

    labels = [
        f"{learning_rate:g}"
        for learning_rate in LEARNING_RATES
    ]

    x = np.arange(
        len(labels)
    )

    plt.figure(figsize=(9, 6))

    plt.bar(
        x,
        means,
        yerr=stds,
        capsize=6,
    )

    plt.axhline(
        y=500,
        linestyle="--",
        linewidth=1,
        label="Maximum reward (500)",
    )

    plt.xticks(
        x,
        labels,
    )

    plt.xlabel("Learning rate")
    plt.ylabel("Final evaluation reward")

    plt.title(
        "REINFORCE on CartPole-v1\n"
        "Final evaluation reward across 5 seeds"
    )

    plt.ylim(0, 550)

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.legend()
    plt.tight_layout()

    output_path = (
        PLOTS_DIR
        / "final_reward_summary.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print("Saved:", output_path)


def plot_training_reward(data):
    plt.figure(figsize=(11, 7))

    for learning_rate in LEARNING_RATES:
        smoothed_runs = []

        for run in data[learning_rate]:
            smoothed_reward = moving_average(
                run["train_rewards"],
                MOVING_AVERAGE_WINDOW,
            )

            smoothed_runs.append(
                smoothed_reward
            )

        rewards = np.stack(
            smoothed_runs
        )

        mean_rewards = rewards.mean(axis=0)
        std_rewards = rewards.std(axis=0)

        original_episodes = (
            data[learning_rate][0][
                "train_episodes"
            ]
        )

        episodes = original_episodes[
            MOVING_AVERAGE_WINDOW - 1:
        ]

        lower = np.clip(
            mean_rewards - std_rewards,
            0,
            500,
        )

        upper = np.clip(
            mean_rewards + std_rewards,
            0,
            500,
        )

        plt.plot(
            episodes,
            mean_rewards,
            linewidth=2,
            label=f"lr={learning_rate:g}",
        )

        plt.fill_between(
            episodes,
            lower,
            upper,
            alpha=0.2,
        )

    plt.axhline(
        y=500,
        linestyle="--",
        linewidth=1,
        label="Maximum reward (500)",
    )

    plt.xlabel("Training episode")

    plt.ylabel(
        f"Training reward "
        f"({MOVING_AVERAGE_WINDOW}-episode moving average)"
    )

    plt.title(
        "REINFORCE on CartPole-v1\n"
        "Training reward across 5 random seeds"
    )

    plt.ylim(0, 520)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = (
        PLOTS_DIR
        / "training_reward_mean_std.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print("Saved:", output_path)


def plot_policy_loss(data):
    plt.figure(figsize=(11, 7))

    for learning_rate in LEARNING_RATES:
        smoothed_runs = []

        for run in data[learning_rate]:
            smoothed_loss = moving_average(
                run["train_losses"],
                MOVING_AVERAGE_WINDOW,
            )

            smoothed_runs.append(
                smoothed_loss
            )

        losses = np.stack(
            smoothed_runs
        )

        mean_losses = losses.mean(axis=0)
        std_losses = losses.std(axis=0)

        original_episodes = (
            data[learning_rate][0][
                "train_episodes"
            ]
        )

        episodes = original_episodes[
            MOVING_AVERAGE_WINDOW - 1:
        ]

        plt.plot(
            episodes,
            mean_losses,
            linewidth=2,
            label=f"lr={learning_rate:g}",
        )

        plt.fill_between(
            episodes,
            mean_losses - std_losses,
            mean_losses + std_losses,
            alpha=0.2,
        )

    plt.xlabel("Training episode")

    plt.ylabel(
        f"Policy loss "
        f"({MOVING_AVERAGE_WINDOW}-episode moving average)"
    )

    plt.title(
        "REINFORCE on CartPole-v1\n"
        "Policy loss across 5 random seeds"
    )

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = (
        PLOTS_DIR
        / "policy_loss_mean_std.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print("Saved:", output_path)


# =====================================================================
# Extended experiment plots
# =====================================================================


def plot_extended_evaluation(runs):
    episodes = runs[0]["eval_episodes"]

    rewards = np.stack(
        [
            run["eval_rewards"]
            for run in runs
        ]
    )

    mean_rewards = rewards.mean(axis=0)
    std_rewards = rewards.std(axis=0)

    lower = np.clip(
        mean_rewards - std_rewards,
        0,
        500,
    )

    upper = np.clip(
        mean_rewards + std_rewards,
        0,
        500,
    )

    plt.figure(figsize=(12, 7))

    plt.plot(
        episodes,
        mean_rewards,
        linewidth=2,
        label="lr=0.001, mean over 5 seeds",
    )

    plt.fill_between(
        episodes,
        lower,
        upper,
        alpha=0.2,
        label="± 1 standard deviation",
    )

    plt.axvline(
        x=1000,
        linestyle="--",
        linewidth=1.5,
        label="Original training budget (1000)",
    )

    plt.axhline(
        y=500,
        linestyle="--",
        linewidth=1,
        label="Maximum reward (500)",
    )

    plt.xlabel("Training episode")
    plt.ylabel("Average evaluation reward")

    plt.title(
        "REINFORCE on CartPole-v1\n"
        "Extended training with lr=0.001"
    )

    plt.ylim(0, 520)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = (
        PLOTS_DIR
        / "extended_evaluation_mean_std.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print("Saved:", output_path)


def plot_extended_individual_runs(runs):
    plt.figure(figsize=(12, 7))

    for run in runs:
        plt.plot(
            run["eval_episodes"],
            run["eval_rewards"],
            linewidth=1.4,
            label=f"seed={run['seed']}",
        )

    plt.axvline(
        x=1000,
        linestyle="--",
        linewidth=1.5,
        label="Original training budget (1000)",
    )

    plt.axhline(
        y=500,
        linestyle="--",
        linewidth=1,
        label="Maximum reward (500)",
    )

    plt.xlabel("Training episode")
    plt.ylabel("Average evaluation reward")

    plt.title(
        "REINFORCE on CartPole-v1\n"
        "Extended lr=0.001 runs across 5 seeds"
    )

    plt.ylim(0, 520)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = (
        PLOTS_DIR
        / "extended_evaluation_individual.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print("Saved:", output_path)


def plot_extended_training_reward(runs):
    smoothed_runs = []

    for run in runs:
        smoothed_reward = moving_average(
            run["train_rewards"],
            MOVING_AVERAGE_WINDOW,
        )

        smoothed_runs.append(
            smoothed_reward
        )

    rewards = np.stack(
        smoothed_runs
    )

    mean_rewards = rewards.mean(axis=0)
    std_rewards = rewards.std(axis=0)

    original_episodes = runs[0][
        "train_episodes"
    ]

    episodes = original_episodes[
        MOVING_AVERAGE_WINDOW - 1:
    ]

    lower = np.clip(
        mean_rewards - std_rewards,
        0,
        500,
    )

    upper = np.clip(
        mean_rewards + std_rewards,
        0,
        500,
    )

    plt.figure(figsize=(12, 7))

    plt.plot(
        episodes,
        mean_rewards,
        linewidth=2,
        label="Mean training reward",
    )

    plt.fill_between(
        episodes,
        lower,
        upper,
        alpha=0.2,
        label="± 1 standard deviation",
    )

    plt.axvline(
        x=1000,
        linestyle="--",
        linewidth=1.5,
        label="Original training budget (1000)",
    )

    plt.axhline(
        y=500,
        linestyle="--",
        linewidth=1,
        label="Maximum reward (500)",
    )

    plt.xlabel("Training episode")

    plt.ylabel(
        f"Training reward "
        f"({MOVING_AVERAGE_WINDOW}-episode moving average)"
    )

    plt.title(
        "REINFORCE on CartPole-v1\n"
        "Extended training reward with lr=0.001"
    )

    plt.ylim(0, 520)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = (
        PLOTS_DIR
        / "extended_training_reward_mean_std.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print("Saved:", output_path)


def plot_extended_policy_loss(runs):
    smoothed_runs = []

    for run in runs:
        smoothed_loss = moving_average(
            run["train_losses"],
            MOVING_AVERAGE_WINDOW,
        )

        smoothed_runs.append(
            smoothed_loss
        )

    losses = np.stack(
        smoothed_runs
    )

    mean_losses = losses.mean(axis=0)
    std_losses = losses.std(axis=0)

    original_episodes = runs[0][
        "train_episodes"
    ]

    episodes = original_episodes[
        MOVING_AVERAGE_WINDOW - 1:
    ]

    plt.figure(figsize=(12, 7))

    plt.plot(
        episodes,
        mean_losses,
        linewidth=2,
        label="Mean policy loss",
    )

    plt.fill_between(
        episodes,
        mean_losses - std_losses,
        mean_losses + std_losses,
        alpha=0.2,
        label="± 1 standard deviation",
    )

    plt.axvline(
        x=1000,
        linestyle="--",
        linewidth=1.5,
        label="Original training budget (1000)",
    )

    plt.xlabel("Training episode")

    plt.ylabel(
        f"Policy loss "
        f"({MOVING_AVERAGE_WINDOW}-episode moving average)"
    )

    plt.title(
        "REINFORCE on CartPole-v1\n"
        "Extended policy loss with lr=0.001"
    )

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = (
        PLOTS_DIR
        / "extended_policy_loss_mean_std.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print("Saved:", output_path)


def plot_1000_vs_2000(runs):
    rewards_1000 = []
    rewards_2000 = []

    for run in runs:
        episodes = run["eval_episodes"]
        rewards = run["eval_rewards"]

        index_1000 = np.where(
            episodes == 1000
        )[0]

        index_2000 = np.where(
            episodes == 2000
        )[0]

        if len(index_1000) == 0:
            raise ValueError(
                f"Episode 1000 not found "
                f"for seed {run['seed']}"
            )

        if len(index_2000) == 0:
            raise ValueError(
                f"Episode 2000 not found "
                f"for seed {run['seed']}"
            )

        rewards_1000.append(
            rewards[index_1000[0]]
        )

        rewards_2000.append(
            rewards[index_2000[0]]
        )

    rewards_1000 = np.array(
        rewards_1000
    )

    rewards_2000 = np.array(
        rewards_2000
    )

    mean_1000 = rewards_1000.mean()
    std_1000 = rewards_1000.std()

    mean_2000 = rewards_2000.mean()
    std_2000 = rewards_2000.std()

    print()
    print(
        "=== Extended training summary ==="
    )

    print(
        "Rewards at episode 1000:",
        [
            round(float(value), 2)
            for value in rewards_1000
        ],
    )

    print(
        "Rewards at episode 2000:",
        [
            round(float(value), 2)
            for value in rewards_2000
        ],
    )

    print(
        f"Episode 1000: "
        f"{mean_1000:.2f} ± {std_1000:.2f}"
    )

    print(
        f"Episode 2000: "
        f"{mean_2000:.2f} ± {std_2000:.2f}"
    )

    print(
        f"Mean improvement: "
        f"{mean_2000 - mean_1000:.2f}"
    )

    means = [
        mean_1000,
        mean_2000,
    ]

    stds = [
        std_1000,
        std_2000,
    ]

    x = np.arange(2)

    plt.figure(figsize=(8, 6))

    plt.bar(
        x,
        means,
        yerr=stds,
        capsize=7,
    )

    plt.xticks(
        x,
        [
            "1000 episodes",
            "2000 episodes",
        ],
    )

    plt.axhline(
        y=500,
        linestyle="--",
        linewidth=1,
        label="Maximum reward (500)",
    )

    plt.ylabel(
        "Evaluation reward"
    )

    plt.title(
        "REINFORCE on CartPole-v1\n"
        "Effect of extended training (lr=0.001)"
    )

    plt.ylim(0, 530)

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.legend()
    plt.tight_layout()

    output_path = (
        PLOTS_DIR
        / "training_budget_1000_vs_2000.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print("Saved:", output_path)


def print_extended_best_summary(runs):
    print()
    print(
        "=== Best performance per extended run ==="
    )

    best_episodes = []

    for run in runs:
        rewards = run["eval_rewards"]
        episodes = run["eval_episodes"]

        best_index = np.argmax(
            rewards
        )

        best_reward = rewards[
            best_index
        ]

        best_episode = episodes[
            best_index
        ]

        best_episodes.append(
            best_episode
        )

        final_reward = rewards[-1]

        print(
            f"Seed {run['seed']}: "
            f"best={best_reward:.2f} "
            f"at episode {best_episode}, "
            f"final={final_reward:.2f}"
        )

    print(
        "Mean first/best checkpoint episode:",
        f"{np.mean(best_episodes):.2f}",
    )


# =====================================================================
# Main
# =====================================================================


def main():
    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "============================================"
    )
    print(
        "STANDARD 1000-EPISODE EXPERIMENTS"
    )
    print(
        "============================================"
    )

    data = load_all_runs()

    plot_evaluation_mean_std(
        data
    )

    plot_individual_evaluation_runs(
        data
    )

    plot_final_reward_summary(
        data
    )

    plot_training_reward(
        data
    )

    plot_policy_loss(
        data
    )

    print()
    print(
        "============================================"
    )
    print(
        "EXTENDED 2000-EPISODE EXPERIMENTS"
    )
    print(
        "============================================"
    )

    extended_runs = load_extended_runs()

    plot_extended_evaluation(
        extended_runs
    )

    plot_extended_individual_runs(
        extended_runs
    )

    plot_extended_training_reward(
        extended_runs
    )

    plot_extended_policy_loss(
        extended_runs
    )

    plot_1000_vs_2000(
        extended_runs
    )

    print_extended_best_summary(
        extended_runs
    )

    print()
    print(
        "============================================"
    )
    print(
        "All plots generated successfully."
    )
    print(
        "============================================"
    )


if __name__ == "__main__":
    main()