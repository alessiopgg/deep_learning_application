import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).parent
RUNS_DIR = BASE_DIR / "runs"
ROBUST_DIR = BASE_DIR / "robust_evaluation"
PLOTS_DIR = BASE_DIR / "plots"


TRAINING_SEEDS = [
    42,
    123,
    456,
    789,
    1000,
]

TRAINING_EPISODES = 2000

POLICY_LEARNING_RATE = 0.001
VALUE_LEARNING_RATE = 0.001


METHODS = {
    "vanilla": "Vanilla REINFORCE",
    "standardized": "Standardized returns",
    "value_baseline": "Value baseline",
}


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


def load_csv(
    path,
):
    with open(
        path,
        newline="",
    ) as file:
        return list(
            csv.DictReader(file)
        )


def load_evaluation_metrics(
    method,
    seed,
):
    run_dir = build_run_dir(
        method,
        seed,
    )

    path = (
        run_dir
        / "evaluation_metrics.csv"
    )

    rows = load_csv(path)

    episodes = np.array(
        [
            int(row["episode"])
            for row in rows
        ],
        dtype=np.int64,
    )

    rewards = np.array(
        [
            float(row["average_reward"])
            for row in rows
        ],
        dtype=np.float64,
    )

    return episodes, rewards


def load_training_metrics(
    method,
    seed,
):
    run_dir = build_run_dir(
        method,
        seed,
    )

    path = (
        run_dir
        / "training_metrics.csv"
    )

    rows = load_csv(path)

    episodes = np.array(
        [
            int(row["episode"])
            for row in rows
        ],
        dtype=np.int64,
    )

    rewards = np.array(
        [
            float(row["reward"])
            for row in rows
        ],
        dtype=np.float64,
    )

    policy_losses = np.array(
        [
            float(row["policy_loss"])
            for row in rows
        ],
        dtype=np.float64,
    )

    value_losses = None

    if (
        method == "value_baseline"
        and
        "value_loss" in rows[0]
    ):
        value_losses = np.array(
            [
                float(row["value_loss"])
                for row in rows
            ],
            dtype=np.float64,
        )

    return (
        episodes,
        rewards,
        policy_losses,
        value_losses,
    )


def moving_average(
    values,
    window,
):
    if window <= 1:
        return values

    kernel = (
        np.ones(window)
        / window
    )

    return np.convolve(
        values,
        kernel,
        mode="valid",
    )


def load_robust_summary():
    path = (
        ROBUST_DIR
        / "checkpoint_summary.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            "Robust evaluation summary not found: "
            f"{path}\n"
            "Run first:\n"
            "python -m Exercise2.evaluate_checkpoints"
        )

    return load_csv(path)


def plot_evaluation_mean_std():
    plt.figure(
        figsize=(11, 6)
    )

    for method, label in METHODS.items():

        all_rewards = []

        episodes = None

        for seed in TRAINING_SEEDS:

            (
                seed_episodes,
                seed_rewards,
            ) = load_evaluation_metrics(
                method,
                seed,
            )

            if episodes is None:
                episodes = seed_episodes

            all_rewards.append(
                seed_rewards
            )

        reward_matrix = np.stack(
            all_rewards
        )

        mean_reward = (
            reward_matrix.mean(axis=0)
        )

        std_reward = (
            reward_matrix.std(axis=0)
        )

        line = plt.plot(
            episodes,
            mean_reward,
            linewidth=2,
            label=label,
        )[0]

        plt.fill_between(
            episodes,
            mean_reward - std_reward,
            mean_reward + std_reward,
            alpha=0.18,
            color=line.get_color(),
        )

    plt.axhline(
        y=500,
        linestyle="--",
        linewidth=1,
        label="Maximum reward (500)",
    )

    plt.xlabel(
        "Training episode"
    )

    plt.ylabel(
        "Average evaluation reward"
    )

    plt.title(
        "Exercise 2 - Evaluation reward across 5 training seeds"
    )

    plt.ylim(
        0,
        520,
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    path = (
        PLOTS_DIR
        / "evaluation_mean_std.png"
    )

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()

    print("Saved:", path)


def plot_evaluation_individual():
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 15),
        sharex=True,
        sharey=True,
    )

    for axis, (
        method,
        label,
    ) in zip(
        axes,
        METHODS.items(),
    ):

        for seed in TRAINING_SEEDS:

            (
                episodes,
                rewards,
            ) = load_evaluation_metrics(
                method,
                seed,
            )

            axis.plot(
                episodes,
                rewards,
                linewidth=1.4,
                label=f"seed {seed}",
            )

        axis.axhline(
            y=500,
            linestyle="--",
            linewidth=1,
        )

        axis.set_title(
            label
        )

        axis.set_ylabel(
            "Evaluation reward"
        )

        axis.set_ylim(
            0,
            520,
        )

        axis.grid(
            alpha=0.3
        )

        axis.legend(
            ncol=5,
            fontsize=8,
        )

    axes[-1].set_xlabel(
        "Training episode"
    )

    fig.suptitle(
        "Exercise 2 - Individual training seeds",
        fontsize=14,
    )

    plt.tight_layout()

    path = (
        PLOTS_DIR
        / "evaluation_individual.png"
    )

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()

    print("Saved:", path)


def plot_training_reward_mean_std():
    window = 50

    plt.figure(
        figsize=(11, 6)
    )

    for method, label in METHODS.items():

        smoothed_rewards = []

        episodes = None

        for seed in TRAINING_SEEDS:

            (
                seed_episodes,
                rewards,
                _,
                _,
            ) = load_training_metrics(
                method,
                seed,
            )

            smoothed = moving_average(
                rewards,
                window,
            )

            if episodes is None:
                episodes = (
                    seed_episodes[
                        window - 1:
                    ]
                )

            smoothed_rewards.append(
                smoothed
            )

        matrix = np.stack(
            smoothed_rewards
        )

        mean_reward = (
            matrix.mean(axis=0)
        )

        std_reward = (
            matrix.std(axis=0)
        )

        line = plt.plot(
            episodes,
            mean_reward,
            linewidth=2,
            label=label,
        )[0]

        plt.fill_between(
            episodes,
            mean_reward - std_reward,
            mean_reward + std_reward,
            alpha=0.18,
            color=line.get_color(),
        )

    plt.axhline(
        y=500,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel(
        "Training episode"
    )

    plt.ylabel(
        f"Training reward ({window}-episode moving average)"
    )

    plt.title(
        "Exercise 2 - Training reward across 5 seeds"
    )

    plt.ylim(
        0,
        520,
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    path = (
        PLOTS_DIR
        / "training_reward_mean_std.png"
    )

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()

    print("Saved:", path)


def plot_value_loss_mean_std():
    window = 50

    all_losses = []
    episodes = None

    for seed in TRAINING_SEEDS:

        (
            seed_episodes,
            _,
            _,
            value_losses,
        ) = load_training_metrics(
            "value_baseline",
            seed,
        )

        if value_losses is None:
            raise RuntimeError(
                "Value loss not found."
            )

        smoothed = moving_average(
            value_losses,
            window,
        )

        if episodes is None:
            episodes = (
                seed_episodes[
                    window - 1:
                ]
            )

        all_losses.append(
            smoothed
        )

    matrix = np.stack(
        all_losses
    )

    mean_loss = (
        matrix.mean(axis=0)
    )

    std_loss = (
        matrix.std(axis=0)
    )

    plt.figure(
        figsize=(11, 6)
    )

    line = plt.plot(
        episodes,
        mean_loss,
        linewidth=2,
        label="Value loss mean",
    )[0]

    plt.fill_between(
        episodes,
        mean_loss - std_loss,
        mean_loss + std_loss,
        alpha=0.18,
        color=line.get_color(),
        label="±1 std across seeds",
    )

    plt.xlabel(
        "Training episode"
    )

    plt.ylabel(
        f"Value MSE ({window}-episode moving average)"
    )

    plt.title(
        "Exercise 2 - ValueNetwork loss across 5 seeds"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    path = (
        PLOTS_DIR
        / "value_loss_mean_std.png"
    )

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()

    print("Saved:", path)


def robust_values_by_method(
    rows,
    checkpoint,
    metric,
):
    result = {}

    for method in METHODS:

        method_rows = [
            row
            for row in rows
            if (
                row["method"] == method
                and
                row["checkpoint"]
                == checkpoint
            )
        ]

        method_rows = sorted(
            method_rows,
            key=lambda row: int(
                row["training_seed"]
            ),
        )

        result[method] = np.array(
            [
                float(row[metric])
                for row in method_rows
            ],
            dtype=np.float64,
        )

    return result


def plot_robust_reward_comparison():
    rows = load_robust_summary()

    methods = list(
        METHODS.keys()
    )

    labels = [
        METHODS[method]
        for method in methods
    ]

    best = robust_values_by_method(
        rows,
        "best",
        "mean_reward",
    )

    final = robust_values_by_method(
        rows,
        "final",
        "mean_reward",
    )

    best_means = np.array(
        [
            best[method].mean()
            for method in methods
        ]
    )

    best_stds = np.array(
        [
            best[method].std()
            for method in methods
        ]
    )

    final_means = np.array(
        [
            final[method].mean()
            for method in methods
        ]
    )

    final_stds = np.array(
        [
            final[method].std()
            for method in methods
        ]
    )

    x = np.arange(
        len(methods)
    )

    width = 0.35

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        x - width / 2,
        best_means,
        width,
        yerr=best_stds,
        capsize=5,
        label="Best checkpoint",
    )

    plt.bar(
        x + width / 2,
        final_means,
        width,
        yerr=final_stds,
        capsize=5,
        label="Final checkpoint",
    )

    plt.xticks(
        x,
        labels,
    )

    plt.ylabel(
        "Mean robust evaluation reward"
    )

    plt.title(
        "Exercise 2 - Robust evaluation over 100 independent episodes"
    )

    plt.ylim(
        450,
        505,
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    path = (
        PLOTS_DIR
        / "robust_reward_comparison.png"
    )

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()

    print("Saved:", path)


def plot_robust_success_rate_comparison():
    rows = load_robust_summary()

    methods = list(
        METHODS.keys()
    )

    labels = [
        METHODS[method]
        for method in methods
    ]

    best = robust_values_by_method(
        rows,
        "best",
        "success_rate_500",
    )

    final = robust_values_by_method(
        rows,
        "final",
        "success_rate_500",
    )

    best_means = np.array(
        [
            100 * best[method].mean()
            for method in methods
        ]
    )

    final_means = np.array(
        [
            100 * final[method].mean()
            for method in methods
        ]
    )

    x = np.arange(
        len(methods)
    )

    width = 0.35

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        x - width / 2,
        best_means,
        width,
        label="Best checkpoint",
    )

    plt.bar(
        x + width / 2,
        final_means,
        width,
        label="Final checkpoint",
    )

    plt.xticks(
        x,
        labels,
    )

    plt.ylabel(
        "Success rate @500 (%)"
    )

    plt.title(
        "Exercise 2 - Robust success rate"
    )

    plt.ylim(
        80,
        101,
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    path = (
        PLOTS_DIR
        / "robust_success_rate_comparison.png"
    )

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()

    print("Saved:", path)


def plot_final_robust_reward_by_seed():
    rows = load_robust_summary()

    x = np.arange(
        len(TRAINING_SEEDS)
    )

    width = 0.25

    plt.figure(
        figsize=(11, 6)
    )

    for index, (
        method,
        label,
    ) in enumerate(
        METHODS.items()
    ):
        values = robust_values_by_method(
            rows,
            "final",
            "mean_reward",
        )[method]

        offset = (
            index - 1
        ) * width

        plt.bar(
            x + offset,
            values,
            width,
            label=label,
        )

    plt.xticks(
        x,
        [
            str(seed)
            for seed in TRAINING_SEEDS
        ],
    )

    plt.xlabel(
        "Training seed"
    )

    plt.ylabel(
        "Mean reward over 100 evaluation episodes"
    )

    plt.title(
        "Exercise 2 - Final checkpoint robustness by training seed"
    )

    plt.ylim(
        450,
        505,
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    path = (
        PLOTS_DIR
        / "final_robust_reward_by_seed.png"
    )

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()

    print("Saved:", path)


def print_summary():
    rows = load_robust_summary()

    print()
    print(
        "============================================================"
    )
    print(
        "FINAL ROBUST COMPARISON"
    )
    print(
        "============================================================"
    )

    for method, label in METHODS.items():

        final_rows = [
            row
            for row in rows
            if (
                row["method"] == method
                and
                row["checkpoint"] == "final"
            )
        ]

        rewards = np.array(
            [
                float(row["mean_reward"])
                for row in final_rows
            ]
        )

        successes = np.array(
            [
                float(
                    row["success_rate_500"]
                )
                for row in final_rows
            ]
        )

        print()
        print(label)

        print(
            "  Reward:",
            f"{rewards.mean():.2f} "
            f"± {rewards.std():.2f}",
        )

        print(
            "  Success @500:",
            f"{100 * successes.mean():.2f}%",
        )


def main():
    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_evaluation_mean_std()

    plot_evaluation_individual()

    plot_training_reward_mean_std()

    plot_value_loss_mean_std()

    plot_robust_reward_comparison()

    plot_robust_success_rate_comparison()

    plot_final_robust_reward_by_seed()

    print_summary()

    print()
    print(
        "All Exercise 2 plots generated successfully."
    )


if __name__ == "__main__":
    main()