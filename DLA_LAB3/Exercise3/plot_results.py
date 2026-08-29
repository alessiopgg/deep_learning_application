import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).parent
RUNS_DIR = BASE_DIR / "runs"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"
SEEDS = [42, 123, 456]


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")

    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def moving_average(values, window):
    if window <= 0:
        raise ValueError("window must be positive")

    averages = []
    running_sum = 0.0

    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]

        current_window = min(index + 1, window)
        averages.append(running_sum / current_window)

    return averages


def run_dir(prefix, seed):
    return RUNS_DIR / f"{prefix}_seed{seed}"


def plot_training_reward(environment, prefix, window, threshold, filename):
    fig, ax = plt.subplots(figsize=(9, 5))

    for seed in SEEDS:
        rows = read_csv(run_dir(prefix, seed) / "training_metrics.csv")
        episodes = [int(row["episode"]) for row in rows]
        rewards = [float(row["reward"]) for row in rows]
        smoothed = moving_average(rewards, window)

        ax.plot(
            episodes,
            smoothed,
            linewidth=1.7,
            label=f"seed {seed}",
        )

    ax.axhline(threshold, linestyle="--", linewidth=1, label=f"reference {threshold:g}")
    ax.set_xlabel("Training episode")
    ax.set_ylabel(f"Reward ({window}-episode moving average)")
    ax.set_title(f"{environment} — Training reward")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=200)
    plt.close(fig)


def plot_evaluation_curve(environment, prefix, threshold, filename, ylim=None):
    fig, ax = plt.subplots(figsize=(9, 5))

    for seed in SEEDS:
        rows = read_csv(run_dir(prefix, seed) / "evaluation_metrics.csv")
        episodes = [int(row["episode"]) for row in rows]
        rewards = [float(row["average_reward"]) for row in rows]

        ax.plot(
            episodes,
            rewards,
            marker="o",
            markersize=2.5,
            linewidth=1.4,
            label=f"seed {seed}",
        )

    ax.axhline(threshold, linestyle="--", linewidth=1, label=f"reference {threshold:g}")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Greedy evaluation mean reward")
    ax.set_title(f"{environment} — Evaluation during training")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=200)
    plt.close(fig)


def environment_summary(environment):
    rows = [
        row
        for row in read_csv(RESULTS_DIR / "final_test_summary.csv")
        if row["environment"] == environment
    ]
    rows.sort(key=lambda row: int(row["training_seed"]))
    return rows


def environment_episodes(environment):
    rows = [
        row
        for row in read_csv(RESULTS_DIR / "final_test_episodes.csv")
        if row["environment"] == environment
    ]
    return rows


def plot_robust_test(environment, threshold, filename, ylim=None):
    rows = environment_summary(environment)
    labels = [row["training_seed"] for row in rows]
    means = [float(row["mean_reward"]) for row in rows]
    stds = [float(row["std_reward"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(labels, means, yerr=stds, capsize=5)
    ax.axhline(threshold, linestyle="--", linewidth=1, label=f"reference {threshold:g}")
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Reward over 100 unseen test episodes")
    ax.set_title(f"{environment} — Robust final test")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=200)
    plt.close(fig)


def plot_reward_distribution(environment, threshold, filename, ylim=None):
    rows = environment_episodes(environment)
    data = []

    for seed in SEEDS:
        rewards = [
            float(row["reward"])
            for row in rows
            if int(row["training_seed"]) == seed
        ]
        data.append(rewards)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.boxplot(data, tick_labels=[str(seed) for seed in SEEDS], showmeans=True)
    ax.axhline(threshold, linestyle="--", linewidth=1, label=f"reference {threshold:g}")
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Episode reward")
    ax.set_title(f"{environment} — Test reward distribution")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=200)
    plt.close(fig)


def plot_td_loss(environment, prefix, window, filename):
    fig, ax = plt.subplots(figsize=(9, 5))

    for seed in SEEDS:
        rows = read_csv(run_dir(prefix, seed) / "training_metrics.csv")
        episodes = []
        losses = []

        for row in rows:
            value = row["mean_loss"]
            if value in ("", "None"):
                continue
            episodes.append(int(row["episode"]))
            losses.append(float(value))

        smoothed = moving_average(losses, window)
        ax.plot(
            episodes,
            smoothed,
            linewidth=1.6,
            label=f"seed {seed}",
        )

    ax.set_xlabel("Training episode")
    ax.set_ylabel(f"Mean TD loss ({window}-episode moving average)")
    ax.set_title(f"{environment} — TD-loss evolution")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=200)
    plt.close(fig)


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Remove only the previous compact plots to avoid stale duplicates.
    for stale_name in [
        "cartpole_training.png",
        "cartpole_test.png",
        "lunarlander_training.png",
        "lunarlander_test.png",
    ]:
        stale_path = PLOTS_DIR / stale_name
        if stale_path.exists():
            stale_path.unlink()

    plot_training_reward(
        "CartPole-v1",
        "cartpole",
        window=25,
        threshold=475,
        filename="cartpole_training_reward.png",
    )
    plot_evaluation_curve(
        "CartPole-v1",
        "cartpole",
        threshold=475,
        filename="cartpole_evaluation_curve.png",
        ylim=(0, 520),
    )
    plot_robust_test(
        "CartPole-v1",
        threshold=475,
        filename="cartpole_robust_test.png",
        ylim=(0, 520),
    )
    plot_reward_distribution(
        "CartPole-v1",
        threshold=475,
        filename="cartpole_selected_reward_distribution.png",
        ylim=(0, 520),
    )

    plot_training_reward(
        "LunarLander-v3",
        "lunarlander",
        window=50,
        threshold=200,
        filename="lunarlander_training_reward.png",
    )
    plot_evaluation_curve(
        "LunarLander-v3",
        "lunarlander",
        threshold=200,
        filename="lunarlander_evaluation_curve.png",
    )
    plot_robust_test(
        "LunarLander-v3",
        threshold=200,
        filename="lunarlander_robust_test.png",
    )
    plot_reward_distribution(
        "LunarLander-v3",
        threshold=200,
        filename="lunarlander_selected_reward_distribution.png",
    )
    plot_td_loss(
        "LunarLander-v3",
        "lunarlander",
        window=25,
        filename="lunarlander_td_loss.png",
    )

    print("Plots saved in:", PLOTS_DIR)


if __name__ == "__main__":
    main()
