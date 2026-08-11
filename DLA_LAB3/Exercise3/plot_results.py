import csv
from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).parent
RUNS_DIR = BASE_DIR / "runs"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"


CARTPOLE_RUNS = {
    "MSE, lr=1e-3": (
        RUNS_DIR
        / "cartpole_dqn_pilot_seed42"
        / "evaluation_metrics.csv"
    ),
    "MSE, lr=5e-4": (
        RUNS_DIR
        / "cartpole_dqn_lr0.0005_seed42"
        / "evaluation_metrics.csv"
    ),
    "Huber, lr=5e-4": (
        RUNS_DIR
        / "cartpole_dqn_huber_lr0.0005_seed42"
        / "evaluation_metrics.csv"
    ),
}


LUNARLANDER_EVAL_PATH = (
    RUNS_DIR
    / "lunarlander_dqn_final_1000ep_seed42"
    / "evaluation_metrics.csv"
)

LUNARLANDER_TRAIN_PATH = (
    RUNS_DIR
    / "lunarlander_dqn_final_1000ep_seed42"
    / "training_metrics.csv"
)

ROBUST_SUMMARY_PATH = (
    RESULTS_DIR
    / "robust_evaluation_summary.csv"
)

ROBUST_EPISODES_PATH = (
    RESULTS_DIR
    / "robust_evaluation_episodes.csv"
)


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing CSV file: {path}"
        )

    with open(
        path,
        newline="",
    ) as file:
        return list(
            csv.DictReader(file)
        )


def moving_average(values, window):
    if window <= 0:
        raise ValueError(
            "window must be positive"
        )

    averages = []

    running_sum = 0.0

    for index, value in enumerate(values):
        running_sum += value

        if index >= window:
            running_sum -= (
                values[index - window]
            )

        current_window = min(
            index + 1,
            window,
        )

        averages.append(
            running_sum / current_window
        )

    return averages


def plot_cartpole_evaluation():
    plt.figure(
        figsize=(9, 5)
    )

    for label, path in CARTPOLE_RUNS.items():
        rows = read_csv(path)

        episodes = [
            int(row["episode"])
            for row in rows
        ]

        rewards = [
            float(row["average_reward"])
            for row in rows
        ]

        plt.plot(
            episodes,
            rewards,
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=label,
        )

    plt.xlabel(
        "Training episode"
    )

    plt.ylabel(
        "Greedy evaluation average reward"
    )

    plt.title(
        "CartPole-v1 — DQN evaluation during training"
    )

    plt.ylim(
        bottom=0,
        top=500,
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    path = (
        PLOTS_DIR
        / "cartpole_evaluation_comparison.png"
    )

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()

    return path


def plot_cartpole_robust_summary():
    rows = read_csv(
        ROBUST_SUMMARY_PATH
    )

    cartpole_rows = [
        row
        for row in rows
        if row["environment"]
        == "CartPole-v1"
    ]

    labels = [
        "MSE\n1e-3\nfinal",
        "MSE\n5e-4\nbest",
        "MSE\n5e-4\nfinal",
        "Huber\n5e-4\nbest",
        "Huber\n5e-4\nfinal",
    ]

    if len(cartpole_rows) != len(labels):
        raise ValueError(
            "Unexpected number of CartPole "
            "summary rows"
        )

    means = [
        float(row["mean_reward"])
        for row in cartpole_rows
    ]

    standard_deviations = [
        float(row["std_reward"])
        for row in cartpole_rows
    ]

    positions = list(
        range(len(labels))
    )

    plt.figure(
        figsize=(9, 5)
    )

    plt.bar(
        positions,
        means,
        yerr=standard_deviations,
        capsize=5,
    )

    plt.xticks(
        positions,
        labels,
    )

    plt.ylabel(
        "Reward over 100 greedy episodes"
    )

    plt.title(
        "CartPole-v1 — Robust checkpoint evaluation"
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    path = (
        PLOTS_DIR
        / "cartpole_robust_comparison.png"
    )

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()

    return path


def plot_lunarlander_evaluation():
    rows = read_csv(
        LUNARLANDER_EVAL_PATH
    )

    episodes = [
        int(row["episode"])
        for row in rows
    ]

    rewards = [
        float(row["average_reward"])
        for row in rows
    ]

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        episodes,
        rewards,
        marker="o",
        markersize=3,
        linewidth=1.5,
    )

    plt.axhline(
        y=0,
        linestyle="--",
        linewidth=1,
        label="Zero reward",
    )

    plt.xlabel(
        "Training episode"
    )

    plt.ylabel(
        "Greedy evaluation average reward"
    )

    plt.title(
        "LunarLander-v3 — DQN evaluation during training"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    path = (
        PLOTS_DIR
        / "lunarlander_evaluation_curve.png"
    )

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()

    return path


def plot_lunarlander_training_reward():
    rows = read_csv(
        LUNARLANDER_TRAIN_PATH
    )

    episodes = [
        int(row["episode"])
        for row in rows
    ]

    rewards = [
        float(row["reward"])
        for row in rows
    ]

    rolling_rewards = moving_average(
        rewards,
        window=50,
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        episodes,
        rewards,
        linewidth=0.7,
        alpha=0.25,
        label="Episode reward",
    )

    plt.plot(
        episodes,
        rolling_rewards,
        linewidth=2,
        label="50-episode moving average",
    )

    plt.axhline(
        y=0,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel(
        "Training episode"
    )

    plt.ylabel(
        "Training reward"
    )

    plt.title(
        "LunarLander-v3 — Training reward"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    path = (
        PLOTS_DIR
        / "lunarlander_training_reward.png"
    )

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()

    return path


def plot_lunarlander_td_loss():
    rows = read_csv(
        LUNARLANDER_TRAIN_PATH
    )

    episodes = []
    losses = []

    for row in rows:
        mean_loss = row["mean_loss"]

        if mean_loss in (
            "",
            "None",
        ):
            continue

        episodes.append(
            int(row["episode"])
        )

        losses.append(
            float(mean_loss)
        )

    rolling_losses = moving_average(
        losses,
        window=25,
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        episodes,
        losses,
        linewidth=0.7,
        alpha=0.25,
        label="Episode mean TD loss",
    )

    plt.plot(
        episodes,
        rolling_losses,
        linewidth=2,
        label="25-episode moving average",
    )

    plt.xlabel(
        "Training episode"
    )

    plt.ylabel(
        "Mean TD loss"
    )

    plt.title(
        "LunarLander-v3 — TD-loss evolution"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    path = (
        PLOTS_DIR
        / "lunarlander_td_loss.png"
    )

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()

    return path


def plot_lunarlander_robust_summary():
    rows = read_csv(
        ROBUST_SUMMARY_PATH
    )

    lunar_rows = [
        row
        for row in rows
        if row["environment"]
        == "LunarLander-v3"
    ]

    labels = [
        "500 ep.\nfinal",
        "1000 ep.\ntraining-best",
        "1000 ep.\nfinal",
    ]

    if len(lunar_rows) != len(labels):
        raise ValueError(
            "Unexpected number of LunarLander "
            "summary rows"
        )

    means = [
        float(row["mean_reward"])
        for row in lunar_rows
    ]

    standard_deviations = [
        float(row["std_reward"])
        for row in lunar_rows
    ]

    positions = list(
        range(len(labels))
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        positions,
        means,
        yerr=standard_deviations,
        capsize=5,
    )

    plt.xticks(
        positions,
        labels,
    )

    plt.axhline(
        y=0,
        linestyle="--",
        linewidth=1,
    )

    plt.ylabel(
        "Reward over 100 greedy episodes"
    )

    plt.title(
        "LunarLander-v3 — Robust checkpoint evaluation"
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    path = (
        PLOTS_DIR
        / "lunarlander_robust_comparison.png"
    )

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()

    return path


def plot_selected_reward_distributions():
    rows = read_csv(
        ROBUST_EPISODES_PATH
    )

    selected_rows = [
        row
        for row in rows
        if row["selected"] == "True"
    ]

    checkpoint_ids = []

    for row in selected_rows:
        checkpoint_id = row[
            "checkpoint_id"
        ]

        if (
            checkpoint_id
            not in checkpoint_ids
        ):
            checkpoint_ids.append(
                checkpoint_id
            )

    if len(checkpoint_ids) != 2:
        raise ValueError(
            "Expected exactly two selected "
            "checkpoints"
        )

    for checkpoint_id in checkpoint_ids:
        checkpoint_rows = [
            row
            for row in selected_rows
            if row["checkpoint_id"]
            == checkpoint_id
        ]

        rewards = [
            float(row["reward"])
            for row in checkpoint_rows
        ]

        environment_name = (
            checkpoint_rows[0][
                "environment"
            ]
        )

        plt.figure(
            figsize=(8, 5)
        )

        plt.hist(
            rewards,
            bins=15,
            edgecolor="black",
            alpha=0.8,
        )

        plt.xlabel(
            "Episode reward"
        )

        plt.ylabel(
            "Number of episodes"
        )

        plt.title(
            f"{environment_name} — "
            f"Selected checkpoint reward distribution"
        )

        plt.grid(
            axis="y",
            alpha=0.3,
        )

        plt.tight_layout()

        if environment_name == "CartPole-v1":
            filename = (
                "cartpole_selected_reward_distribution.png"
            )
        else:
            filename = (
                "lunarlander_selected_reward_distribution.png"
            )

        path = (
            PLOTS_DIR
            / filename
        )

        plt.savefig(
            path,
            dpi=200,
        )

        plt.close()


def main():
    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    created_paths = []

    created_paths.append(
        plot_cartpole_evaluation()
    )

    created_paths.append(
        plot_cartpole_robust_summary()
    )

    created_paths.append(
        plot_lunarlander_evaluation()
    )

    created_paths.append(
        plot_lunarlander_training_reward()
    )

    created_paths.append(
        plot_lunarlander_td_loss()
    )

    created_paths.append(
        plot_lunarlander_robust_summary()
    )

    plot_selected_reward_distributions()

    created_paths.extend(
        [
            PLOTS_DIR
            / "cartpole_selected_reward_distribution.png",

            PLOTS_DIR
            / "lunarlander_selected_reward_distribution.png",
        ]
    )

    print("Plots created:")

    for path in created_paths:
        print(
            "-",
            path,
        )


if __name__ == "__main__":
    main()
