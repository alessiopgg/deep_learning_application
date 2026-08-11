import csv
import json
import statistics
from pathlib import Path

import gymnasium as gym
import torch

from Exercise3.dqn import QNetwork


NUM_EVAL_EPISODES = 100
BASE_SEED = 1000

BASE_DIR = Path(__file__).parent
RUNS_DIR = BASE_DIR / "runs"
RESULTS_DIR = BASE_DIR / "results"


CHECKPOINTS = [
    {
        "id": "cartpole_mse_lr0.001_final",
        "run_dir": "cartpole_dqn_pilot_seed42",
        "checkpoint": "q_network.pt",
        "checkpoint_role": "final",
        "selected": False,
    },
    {
        "id": "cartpole_mse_lr0.0005_best",
        "run_dir": "cartpole_dqn_lr0.0005_seed42",
        "checkpoint": "best_q_network.pt",
        "checkpoint_role": "best_training_evaluation",
        "selected": True,
    },
    {
        "id": "cartpole_mse_lr0.0005_final",
        "run_dir": "cartpole_dqn_lr0.0005_seed42",
        "checkpoint": "final_q_network.pt",
        "checkpoint_role": "final",
        "selected": False,
    },
    {
        "id": "cartpole_huber_lr0.0005_best",
        "run_dir": "cartpole_dqn_huber_lr0.0005_seed42",
        "checkpoint": "best_q_network.pt",
        "checkpoint_role": "best_training_evaluation",
        "selected": False,
    },
    {
        "id": "cartpole_huber_lr0.0005_final",
        "run_dir": "cartpole_dqn_huber_lr0.0005_seed42",
        "checkpoint": "final_q_network.pt",
        "checkpoint_role": "final",
        "selected": False,
    },
    {
        "id": "lunarlander_500_final",
        "run_dir": "lunarlander_dqn_pilot_seed42",
        "checkpoint": "final_q_network.pt",
        "checkpoint_role": "final",
        "selected": False,
    },
    {
        "id": "lunarlander_1000_best",
        "run_dir": "lunarlander_dqn_final_1000ep_seed42",
        "checkpoint": "best_q_network.pt",
        "checkpoint_role": "best_training_evaluation",
        "selected": False,
    },
    {
        "id": "lunarlander_1000_final",
        "run_dir": "lunarlander_dqn_final_1000ep_seed42",
        "checkpoint": "final_q_network.pt",
        "checkpoint_role": "final",
        "selected": True,
    },
]


def load_config(run_dir):
    config_path = run_dir / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing config file: {config_path}"
        )

    with open(config_path) as file:
        return json.load(file)


def evaluate_checkpoint(
    checkpoint_path,
    environment_name,
    hidden_dim,
):
    env = gym.make(environment_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    network = QNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
    )

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    network.load_state_dict(
        state_dict
    )

    network.eval()

    episode_results = []

    with torch.inference_mode():
        for episode_index in range(
            NUM_EVAL_EPISODES
        ):
            seed = (
                BASE_SEED
                + episode_index
            )

            state, info = env.reset(
                seed=seed
            )

            terminated = False
            truncated = False

            total_reward = 0.0
            episode_length = 0

            while not (
                terminated or truncated
            ):
                state_tensor = torch.as_tensor(
                    state,
                    dtype=torch.float32,
                )

                q_values = network(
                    state_tensor
                )

                action = int(
                    torch.argmax(
                        q_values
                    ).item()
                )

                (
                    state,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action)

                total_reward += reward
                episode_length += 1

            episode_results.append(
                {
                    "episode":
                        episode_index + 1,
                    "seed":
                        seed,
                    "reward":
                        total_reward,
                    "length":
                        episode_length,
                }
            )

    env.close()

    return episode_results


def compute_summary(episode_results):
    rewards = [
        row["reward"]
        for row in episode_results
    ]

    lengths = [
        row["length"]
        for row in episode_results
    ]

    positive_count = sum(
        reward > 0
        for reward in rewards
    )

    reward_ge_100_count = sum(
        reward >= 100
        for reward in rewards
    )

    reward_ge_200_count = sum(
        reward >= 200
        for reward in rewards
    )

    return {
        "mean_reward":
            statistics.mean(rewards),

        "std_reward":
            statistics.pstdev(rewards),

        "median_reward":
            statistics.median(rewards),

        "min_reward":
            min(rewards),

        "max_reward":
            max(rewards),

        "mean_episode_length":
            statistics.mean(lengths),

        "positive_count":
            positive_count,

        "positive_percentage":
            100.0
            * positive_count
            / len(rewards),

        "reward_ge_100_count":
            reward_ge_100_count,

        "reward_ge_100_percentage":
            100.0
            * reward_ge_100_count
            / len(rewards),

        "reward_ge_200_count":
            reward_ge_200_count,

        "reward_ge_200_percentage":
            100.0
            * reward_ge_200_count
            / len(rewards),
    }


def save_episode_results(rows):
    path = (
        RESULTS_DIR
        / "robust_evaluation_episodes.csv"
    )

    fieldnames = [
        "checkpoint_id",
        "environment",
        "selected",
        "episode",
        "seed",
        "reward",
        "length",
    ]

    with open(
        path,
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)

    return path


def save_summary(rows):
    path = (
        RESULTS_DIR
        / "robust_evaluation_summary.csv"
    )

    fieldnames = [
        "checkpoint_id",
        "environment",
        "checkpoint_role",
        "selected",
        "training_episodes",
        "learning_rate",
        "loss",
        "mean_reward",
        "std_reward",
        "median_reward",
        "min_reward",
        "max_reward",
        "mean_episode_length",
        "positive_count",
        "positive_percentage",
        "reward_ge_100_count",
        "reward_ge_100_percentage",
        "reward_ge_200_count",
        "reward_ge_200_percentage",
    ]

    with open(
        path,
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)

    return path


def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Robust evaluation"
    )

    print(
        "Episodes per checkpoint:",
        NUM_EVAL_EPISODES,
    )

    print(
        "Seeds:",
        f"{BASE_SEED}-"
        f"{BASE_SEED + NUM_EVAL_EPISODES - 1}",
    )

    all_episode_rows = []
    summary_rows = []

    for specification in CHECKPOINTS:
        run_dir = (
            RUNS_DIR
            / specification["run_dir"]
        )

        checkpoint_path = (
            run_dir
            / specification["checkpoint"]
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint: "
                f"{checkpoint_path}"
            )

        config = load_config(
            run_dir
        )

        environment_name = (
            config["environment"]
        )

        hidden_dim = (
            config["hidden_dim"]
        )

        print(
            "\nEvaluating:",
            specification["id"],
        )

        episode_results = (
            evaluate_checkpoint(
                checkpoint_path=
                    checkpoint_path,
                environment_name=
                    environment_name,
                hidden_dim=
                    hidden_dim,
            )
        )

        summary = compute_summary(
            episode_results
        )

        for row in episode_results:
            all_episode_rows.append(
                {
                    "checkpoint_id":
                        specification["id"],
                    "environment":
                        environment_name,
                    "selected":
                        specification["selected"],
                    **row,
                }
            )

        summary_rows.append(
            {
                "checkpoint_id":
                    specification["id"],

                "environment":
                    environment_name,

                "checkpoint_role":
                    specification[
                        "checkpoint_role"
                    ],

                "selected":
                    specification["selected"],

                "training_episodes":
                    config["num_episodes"],

                "learning_rate":
                    config["learning_rate"],

                "loss":
                    config["loss"],

                **summary,
            }
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
            f"Min / max: "
            f"{summary['min_reward']:.2f} / "
            f"{summary['max_reward']:.2f}"
        )

        print(
            f"Positive episodes: "
            f"{summary['positive_percentage']:.1f}%"
        )

        print(
            f"Reward >= 100: "
            f"{summary['reward_ge_100_percentage']:.1f}%"
        )

        print(
            f"Reward >= 200: "
            f"{summary['reward_ge_200_percentage']:.1f}%"
        )

    episode_path = save_episode_results(
        all_episode_rows
    )

    summary_path = save_summary(
        summary_rows
    )

    print(
        "\nEvaluation completed."
    )

    print(
        "Episode-level results:",
        episode_path,
    )

    print(
        "Summary:",
        summary_path,
    )


if __name__ == "__main__":
    main()
