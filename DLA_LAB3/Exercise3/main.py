import argparse
import csv
import json
import random
from pathlib import Path

import gymnasium as gym
import torch

from Exercise3.dqn import QNetwork, ReplayBuffer, train_dqn


CONFIG = {
    "config_id": "cartpole_final",
    "environment": "CartPole-v1",
    "max_episodes": 1000,
    "gamma": 0.99,
    "learning_rate": 3e-4,
    "hidden_dims": [128, 128],
    "loss": "huber",
    "buffer_capacity": 50_000,
    "batch_size": 128,
    "min_buffer_size": 1_000,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_steps": 15_000,
    "train_frequency": 4,
    "target_update": "soft",
    "target_tau": 0.005,
    "target_sync_every": 1_000,
    "gradient_clip_norm": 10.0,
    "eval_every": 10,
    "monitor_seed_start": 2_100,
    "monitor_episodes": 20,
    "early_stopping_reward": 475.0,
    "early_stopping_patience": 3,
}

BASE_DIR = Path(__file__).parent
RUNS_DIR = BASE_DIR / "runs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Final DQN training on CartPole-v1"
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_name = f"cartpole_seed{args.seed}"
    output_dir = RUNS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    monitor_seeds = list(
        range(
            CONFIG["monitor_seed_start"],
            CONFIG["monitor_seed_start"]
            + CONFIG["monitor_episodes"],
        )
    )

    train_env = gym.make(CONFIG["environment"])
    eval_env = gym.make(CONFIG["environment"])
    train_env.action_space.seed(args.seed)

    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.n

    online_network = QNetwork(
        state_dim,
        action_dim,
        CONFIG["hidden_dims"],
    )
    target_network = QNetwork(
        state_dim,
        action_dim,
        CONFIG["hidden_dims"],
    )

    optimizer = torch.optim.Adam(
        online_network.parameters(),
        lr=CONFIG["learning_rate"],
    )
    replay_buffer = ReplayBuffer(CONFIG["buffer_capacity"])

    print("Environment:", CONFIG["environment"])
    print("Training seed:", args.seed)
    print("Run:", run_name)

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
        num_episodes=CONFIG["max_episodes"],
        gamma=CONFIG["gamma"],
        batch_size=CONFIG["batch_size"],
        min_buffer_size=CONFIG["min_buffer_size"],
        epsilon_start=CONFIG["epsilon_start"],
        epsilon_end=CONFIG["epsilon_end"],
        epsilon_decay_steps=CONFIG["epsilon_decay_steps"],
        train_frequency=CONFIG["train_frequency"],
        target_update_mode=CONFIG["target_update"],
        target_sync_every=CONFIG["target_sync_every"],
        target_tau=CONFIG["target_tau"],
        eval_every=CONFIG["eval_every"],
        eval_seeds=monitor_seeds,
        checkpoint_path=output_dir / "best_q_network.pt",
        loss_type=CONFIG["loss"],
        gradient_clip_norm=CONFIG["gradient_clip_norm"],
        training_seed=args.seed,
        early_stopping_reward=CONFIG["early_stopping_reward"],
        early_stopping_patience=CONFIG["early_stopping_patience"],
    )

    train_env.close()
    eval_env.close()

    torch.save(
        online_network.state_dict(),
        output_dir / "final_q_network.pt",
    )

    completed_episodes = len(training_history)

    config = {
        **CONFIG,
        "run_name": run_name,
        "seed": args.seed,
        "num_episodes": completed_episodes,
        "early_stopped": completed_episodes < CONFIG["max_episodes"],
        "environment_steps": total_steps,
        "optimizer_updates": updates,
        "checkpoint_selection": (
            "best checkpoint on fixed monitor seeds 2100-2119; "
            "best/final resolved by evaluate_results.py on validation seeds"
        ),
    }

    with open(output_dir / "config.json", "w") as file:
        json.dump(config, file, indent=4)

    save_csv(
        output_dir / "training_metrics.csv",
        training_history,
    )
    save_csv(
        output_dir / "evaluation_metrics.csv",
        evaluation_history,
    )

    print("\nTraining completed")
    print("Episodes:", completed_episodes)
    print("Environment steps:", total_steps)
    print("Optimizer updates:", updates)
    print("Artifacts:", output_dir)


if __name__ == "__main__":
    main()
