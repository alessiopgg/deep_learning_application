import argparse
import csv
import json
import random
from pathlib import Path

import gymnasium as gym
import torch

from Exercise3.dqn import QNetwork, ReplayBuffer, train_dqn


CONFIG = {
    "config_id": "lunarlander_final",
    "environment": "LunarLander-v3",
    "episodes": 2000,
    "gamma": 0.99,
    "learning_rate": 5e-4,
    "hidden_dims": [128, 128],
    "loss": "mse",
    "buffer_capacity": 100_000,
    "batch_size": 64,
    "min_buffer_size": 5_000,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_steps": 100_000,
    "train_frequency": 1,
    "target_update": "hard",
    "target_sync_every": 1_000,
    "target_tau": 0.005,
    "gradient_clip_norm": 10.0,
    "eval_every": 50,
    "monitor_seed_start": 2_000,
    "monitor_episodes": 20,
}

BASE_DIR = Path(__file__).parent
RUNS_DIR = BASE_DIR / "runs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Final DQN training on LunarLander-v3"
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

    run_name = f"lunarlander_seed{args.seed}"
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
        num_episodes=CONFIG["episodes"],
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
    )

    train_env.close()
    eval_env.close()

    torch.save(
        online_network.state_dict(),
        output_dir / "final_q_network.pt",
    )

    config = {
        **CONFIG,
        "run_name": run_name,
        "seed": args.seed,
        "num_episodes": len(training_history),
        "environment_steps": total_steps,
        "optimizer_updates": updates,
        "checkpoint_selection": (
            "best checkpoint on fixed monitor seeds 2000-2019; "
            "best/final resolved by evaluate_results.py"
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
    print("Episodes:", len(training_history))
    print("Environment steps:", total_steps)
    print("Optimizer updates:", updates)
    print("Artifacts:", output_dir)


if __name__ == "__main__":
    main()
