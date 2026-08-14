import argparse
import csv
import json
from pathlib import Path

import gymnasium as gym
import torch

from models import PolicyNetwork
from Exercise2.reinforce_ex2 import train


# Reference configuration derived from Exercise 1 experiments
DEFAULT_SEED = 42
DEFAULT_NUM_EPISODES = 2000
DEFAULT_GAMMA = 0.99

DEFAULT_LEARNING_RATE = 0.001
DEFAULT_HIDDEN_DIM = 64

DEFAULT_EVAL_EVERY = 25
DEFAULT_EVAL_EPISODES = 20


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train REINFORCE on CartPole-v1 "
            "with optional return standardization"
        )
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "vanilla",
            "standardized",
        ],
        default="standardized",
        help=(
            "Training variant: vanilla or standardized "
            "(default: standardized)"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_NUM_EPISODES,
        help=(
            f"Number of training episodes "
            f"(default: {DEFAULT_NUM_EPISODES})"
        ),
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=DEFAULT_GAMMA,
        help=f"Discount factor (default: {DEFAULT_GAMMA})",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=(
            f"Policy learning rate "
            f"(default: {DEFAULT_LEARNING_RATE})"
        ),
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=DEFAULT_HIDDEN_DIM,
        help=(
            f"Hidden layer size "
            f"(default: {DEFAULT_HIDDEN_DIM})"
        ),
    )

    parser.add_argument(
        "--eval-every",
        type=int,
        default=DEFAULT_EVAL_EVERY,
        help=(
            f"Evaluate every N training episodes "
            f"(default: {DEFAULT_EVAL_EVERY})"
        ),
    )

    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=DEFAULT_EVAL_EPISODES,
        help=(
            f"Number of evaluation episodes "
            f"(default: {DEFAULT_EVAL_EPISODES})"
        ),
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional custom name for the run",
    )

    return parser.parse_args()


def build_run_name(args):
    if args.run_name is not None:
        return args.run_name

    return (
        f"reinforce_{args.mode}"
        f"_lr{args.lr:g}"
        f"_gamma{args.gamma:g}"
        f"_h{args.hidden_dim}"
        f"_seed{args.seed}"
    )


def save_results(
    episode_rewards,
    policy_losses,
    evaluation_history,
    policy,
    best_checkpoint,
    args,
    run_name,
):
    output_dir = (
        Path(__file__).parent
        / "runs"
        / run_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    standardize = args.mode == "standardized"

    config = {
        "run_name": run_name,
        "method": args.mode,
        "seed": args.seed,
        "num_episodes": args.episodes,
        "gamma": args.gamma,
        "optimizer": "Adam",
        "learning_rate": args.lr,
        "eval_every": args.eval_every,
        "eval_episodes": args.eval_episodes,
        "hidden_dim": args.hidden_dim,
        "activation": "ReLU",
        "standardize_returns": standardize,
        "value_baseline": False,
        "best_evaluation_episode": (
            best_checkpoint["episode"]
        ),
        "best_evaluation_reward": (
            best_checkpoint["average_reward"]
        ),
    }

    with open(
        output_dir / "config.json",
        "w",
    ) as file:
        json.dump(
            config,
            file,
            indent=4,
        )

    with open(
        output_dir / "training_metrics.csv",
        "w",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "episode",
                "reward",
                "policy_loss",
            ]
        )

        for episode, (
            reward,
            policy_loss,
        ) in enumerate(
            zip(
                episode_rewards,
                policy_losses,
            ),
            start=1,
        ):
            writer.writerow(
                [
                    episode,
                    reward,
                    policy_loss,
                ]
            )

    with open(
        output_dir / "evaluation_metrics.csv",
        "w",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "episode",
                "average_reward",
                "average_length",
            ]
        )

        for evaluation in evaluation_history:
            writer.writerow(
                [
                    evaluation["episode"],
                    evaluation["average_reward"],
                    evaluation["average_length"],
                ]
            )

    # Final policy
    torch.save(
        policy.state_dict(),
        output_dir / "policy.pt",
    )

    # Best periodic-evaluation policy
    torch.save(
        best_checkpoint["state_dict"],
        output_dir / "best_policy.pt",
    )


def main():
    args = parse_args()

    if args.eval_every > args.episodes:
        raise ValueError(
            "--eval-every must be less than or equal "
            "to --episodes."
        )

    run_name = build_run_name(args)

    standardize = args.mode == "standardized"

    torch.manual_seed(args.seed)

    train_env = gym.make(
        "CartPole-v1"
    )

    eval_env = gym.make(
        "CartPole-v1"
    )

    train_env.reset(
        seed=args.seed
    )

    eval_env.reset(
        seed=args.seed + 1
    )

    policy = PolicyNetwork(
        hidden_dim=args.hidden_dim,
    )

    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=args.lr,
    )

    print("Run:", run_name)
    print("Method:", args.mode)
    print("Standardize returns:", standardize)
    print("Seed:", args.seed)
    print("Learning rate:", args.lr)
    print("Gamma:", args.gamma)
    print("Hidden dimension:", args.hidden_dim)
    print("Training episodes:", args.episodes)
    print(
        f"Evaluation: every {args.eval_every} episodes "
        f"for {args.eval_episodes} episodes"
    )
    print()

    (
        episode_rewards,
        policy_losses,
        evaluation_history,
        best_checkpoint,
    ) = train(
        env=train_env,
        eval_env=eval_env,
        policy=policy,
        optimizer=optimizer,
        num_episodes=args.episodes,
        gamma=args.gamma,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        standardize=standardize,
    )

    train_env.close()
    eval_env.close()

    save_results(
        episode_rewards=episode_rewards,
        policy_losses=policy_losses,
        evaluation_history=evaluation_history,
        policy=policy,
        best_checkpoint=best_checkpoint,
        args=args,
        run_name=run_name,
    )

    print()
    print("Training completed")
    print("Run:", run_name)
    print(
        "Training episodes:",
        len(episode_rewards),
    )
    print(
        "Evaluations:",
        len(evaluation_history),
    )
    print(
        "Best evaluation:",
        f"{best_checkpoint['average_reward']:.2f}",
        f"at episode {best_checkpoint['episode']}",
    )


if __name__ == "__main__":
    main()