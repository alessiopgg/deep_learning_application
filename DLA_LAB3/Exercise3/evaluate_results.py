import csv
import json
import shutil
import statistics
from pathlib import Path

import gymnasium as gym
import torch

from Exercise3.dqn import QNetwork


BASE_DIR = Path(__file__).parent
RUNS_DIR = BASE_DIR / "runs"
RESULTS_DIR = BASE_DIR / "results"

TRAINING_SEEDS = [42, 123, 456]
VALIDATION_SEEDS = list(range(2000, 2050))
TEST_SEEDS = list(range(1000, 1100))

RUNS = [
    ("CartPole-v1", seed, f"cartpole_seed{seed}")
    for seed in TRAINING_SEEDS
] + [
    ("LunarLander-v3", seed, f"lunarlander_seed{seed}")
    for seed in TRAINING_SEEDS
]


def save_csv(path, rows):
    if not rows:
        return

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    rewards = [row["reward"] for row in rows]

    return {
        "mean_reward": statistics.mean(rewards),
        "std_reward": statistics.pstdev(rewards),
        "median_reward": statistics.median(rewards),
        "min_reward": min(rewards),
        "max_reward": max(rewards),
    }


def evaluate_checkpoint(run_dir, checkpoint_path, seeds):
    with open(run_dir / "config.json") as file:
        config = json.load(file)

    env = gym.make(config["environment"])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    network = QNetwork(
        state_dim,
        action_dim,
        config["hidden_dims"],
    )
    network.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    )
    network.eval()

    rows = []

    with torch.inference_mode():
        for index, seed in enumerate(seeds, start=1):
            state, _ = env.reset(seed=seed)
            terminated = False
            truncated = False
            total_reward = 0.0
            length = 0

            while not (terminated or truncated):
                state_tensor = torch.as_tensor(
                    state,
                    dtype=torch.float32,
                )
                action = int(network(state_tensor).argmax().item())

                (
                    state,
                    reward,
                    terminated,
                    truncated,
                    _,
                ) = env.step(action)

                total_reward += reward
                length += 1

            rows.append(
                {
                    "episode": index,
                    "seed": seed,
                    "reward": total_reward,
                    "length": length,
                }
            )

    env.close()
    return config, rows


def choose_checkpoint(run_dir):
    candidates = []

    for role in ("best", "final"):
        path = run_dir / f"{role}_q_network.pt"

        if not path.exists():
            continue

        config, rows = evaluate_checkpoint(
            run_dir,
            path,
            VALIDATION_SEEDS,
        )
        summary = summarize(rows)

        candidates.append(
            {
                "role": role,
                "path": path,
                "config": config,
                "summary": summary,
            }
        )

    if candidates:
        selected = max(
            candidates,
            key=lambda item: (
                item["summary"]["mean_reward"],
                item["summary"]["median_reward"],
                -item["summary"]["std_reward"],
            ),
        )

        shutil.copy2(
            selected["path"],
            run_dir / "selected_q_network.pt",
        )

        return (
            selected["config"],
            selected["role"],
            selected["summary"],
        )

    selected_path = run_dir / "selected_q_network.pt"

    if not selected_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found in {run_dir}"
        )

    config, rows = evaluate_checkpoint(
        run_dir,
        selected_path,
        VALIDATION_SEEDS,
    )
    summary = summarize(rows)

    return (
        config,
        config.get("selected_checkpoint_role", "selected"),
        summary,
    )


def success_percentage(environment, rewards):
    threshold = 475.0 if environment == "CartPole-v1" else 200.0
    return (
        threshold,
        100.0
        * sum(reward >= threshold for reward in rewards)
        / len(rewards),
    )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    selection_rows = []
    episode_rows = []
    summary_rows = []

    for environment, training_seed, run_name in RUNS:
        run_dir = RUNS_DIR / run_name

        if not run_dir.exists():
            raise FileNotFoundError(f"Missing run: {run_dir}")

        config, role, validation = choose_checkpoint(run_dir)

        selection_rows.append(
            {
                "environment": environment,
                "training_seed": training_seed,
                "run_name": run_name,
                "checkpoint_role": role,
                "validation_mean_reward": validation["mean_reward"],
                "validation_std_reward": validation["std_reward"],
                "validation_median_reward": validation["median_reward"],
            }
        )

        _, test_rows = evaluate_checkpoint(
            run_dir,
            run_dir / "selected_q_network.pt",
            TEST_SEEDS,
        )
        test_summary = summarize(test_rows)
        rewards = [row["reward"] for row in test_rows]
        threshold, success = success_percentage(
            environment,
            rewards,
        )

        perfect_500 = (
            100.0
            * sum(reward == 500 for reward in rewards)
            / len(rewards)
            if environment == "CartPole-v1"
            else ""
        )

        for row in test_rows:
            episode_rows.append(
                {
                    "environment": environment,
                    "training_seed": training_seed,
                    **row,
                }
            )

        summary_rows.append(
            {
                "environment": environment,
                "training_seed": training_seed,
                "checkpoint_role": role,
                **test_summary,
                "success_threshold": threshold,
                "success_percentage": success,
                "perfect_500_percentage": perfect_500,
            }
        )

        print(
            f"{environment} seed {training_seed}: "
            f"{test_summary['mean_reward']:.2f} "
            f"± {test_summary['std_reward']:.2f}"
        )

    aggregate_rows = []

    for environment in ("CartPole-v1", "LunarLander-v3"):
        rows = [
            row
            for row in summary_rows
            if row["environment"] == environment
        ]
        seed_means = [row["mean_reward"] for row in rows]
        successes = [row["success_percentage"] for row in rows]

        aggregate_rows.append(
            {
                "environment": environment,
                "training_seed_count": len(rows),
                "test_episodes_per_seed": len(TEST_SEEDS),
                "mean_of_seed_means": statistics.mean(seed_means),
                "std_of_seed_means": statistics.pstdev(seed_means),
                "min_seed_mean": min(seed_means),
                "max_seed_mean": max(seed_means),
                "success_threshold": rows[0]["success_threshold"],
                "mean_success_percentage": statistics.mean(successes),
                "mean_perfect_500_percentage": (
                    statistics.mean(
                        row["perfect_500_percentage"]
                        for row in rows
                    )
                    if environment == "CartPole-v1"
                    else ""
                ),
            }
        )

    save_csv(
        RESULTS_DIR / "validation_selection.csv",
        selection_rows,
    )
    save_csv(
        RESULTS_DIR / "final_test_episodes.csv",
        episode_rows,
    )
    save_csv(
        RESULTS_DIR / "final_test_summary.csv",
        summary_rows,
    )
    save_csv(
        RESULTS_DIR / "final_test_aggregate.csv",
        aggregate_rows,
    )

    print("\nFinal aggregate")
    for row in aggregate_rows:
        print(
            f"{row['environment']}: "
            f"{row['mean_of_seed_means']:.2f} "
            f"± {row['std_of_seed_means']:.2f}"
        )


if __name__ == "__main__":
    main()
