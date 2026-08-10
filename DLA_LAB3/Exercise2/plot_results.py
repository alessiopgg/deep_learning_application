import csv
from pathlib import Path
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).parent

RUNS = {
    "Vanilla REINFORCE": (
        BASE_DIR
        / "runs"
        / "no_standardization_seed42"
    ),
    "Standardized returns": (
        BASE_DIR
        / "runs"
        / "standardized_returns_seed42"
    ),
    "Value baseline": (
        BASE_DIR
        / "runs"
        / "value_baseline_seed42"
    ),
}


def load_evaluation_metrics(run_dir):
    episodes = []
    average_rewards = []
    average_lengths = []

    csv_path = run_dir / "evaluation_metrics.csv"

    with open(csv_path, newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            episodes.append(
                int(row["episode"])
            )
            average_rewards.append(
                float(row["average_reward"])
            )
            average_lengths.append(
                float(row["average_length"])
            )

    return (
        episodes,
        average_rewards,
        average_lengths,
    )


def main():
    output_dir = BASE_DIR / "plots"
    output_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(11, 6))

    for label, run_dir in RUNS.items():
        (
            episodes,
            average_rewards,
            _,
        ) = load_evaluation_metrics(
            run_dir
        )

        plt.plot(
            episodes,
            average_rewards,
            marker="o",
            markersize=3,
            label=label,
        )

    plt.axhline(
        y=500,
        linestyle="--",
        linewidth=1,
        label="Maximum CartPole reward (500)",
    )

    plt.xlabel("Training episode")
    plt.ylabel("Average evaluation reward")

    plt.title(
        "REINFORCE on CartPole-v1 - "
        "Exercise 2 Comparison"
    )

    plt.ylim(0, 520)

    plt.grid(
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        output_dir
        / "evaluation_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print(
        "Saved:",
        output_path,
    )


if __name__ == "__main__":
    main()
