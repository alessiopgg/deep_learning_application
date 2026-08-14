import argparse
import csv
import json
from pathlib import Path

import gymnasium as gym
import torch

from models import (
    PolicyNetwork,
    ValueNetwork,
)

from Exercise2.reinforce_ex2 import (
    train_with_value_baseline,
)


DEFAULT_SEED = 42

DEFAULT_NUM_EPISODES = 2000
DEFAULT_GAMMA = 0.99

DEFAULT_POLICY_LEARNING_RATE = 0.001
DEFAULT_VALUE_LEARNING_RATE = 0.001

DEFAULT_HIDDEN_DIM = 64

DEFAULT_EVAL_EVERY = 25
DEFAULT_EVAL_EPISODES = 20


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train REINFORCE with a learned "
            "value baseline on CartPole-v1"
        )
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
        "--policy-lr",
        type=float,
        default=DEFAULT_POLICY_LEARNING_RATE,
        help=(
            f"Policy learning rate "
            f"(default: {DEFAULT_POLICY_LEARNING_RATE})"
        ),
    )

    parser.add_argument(
        "--value-lr",
        type=float,
        default=DEFAULT_VALUE_LEARNING_RATE,
        help=(
            f"Value-network learning rate "
            f"(default: {DEFAULT_VALUE_LEARNING_RATE})"
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
        f"reinforce_value_baseline"
        f"_plr{args.policy_lr:g}"
        f"_vlr{args.value_lr:g}"
        f"_gamma{args.gamma:g}"
        f"_h{args.hidden_dim}"
        f"_seed{args.seed}"
    )


def save_results(
    episode_rewards,
    policy_losses,
    value_losses,
    evaluation_history,
    policy,
    value_network,
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

    config = {
        "run_name": run_name,
        "method": "value_baseline",
        "seed": args.seed,
        "num_episodes": args.episodes,
        "gamma": args.gamma,

        "policy_architecture": (
            f"4-{args.hidden_dim}-2"
        ),
        "policy_activation": "ReLU",
        "policy_optimizer": "Adam",
        "policy_learning_rate": args.policy_lr,

        "value_architecture": (
            f"4-{args.hidden_dim}-1"
        ),
        "value_activation": "ReLU",
        "value_optimizer": "Adam",
        "value_learning_rate": args.value_lr,

        "hidden_dim": args.hidden_dim,

        "eval_every": args.eval_every,
        "eval_episodes": args.eval_episodes,

        "standardize_returns": False,
        "value_baseline": True,

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
                "value_loss",
            ]
        )

        for episode, (
            reward,
            policy_loss,
            value_loss,
        ) in enumerate(
            zip(
                episode_rewards,
                policy_losses,
                value_losses,
            ),
            start=1,
        ):
            writer.writerow(
                [
                    episode,
                    reward,
                    policy_loss,
                    value_loss,
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

    # Final checkpoints
    torch.save(
        policy.state_dict(),
        output_dir / "policy.pt",
    )

    torch.save(
        value_network.state_dict(),
        output_dir / "value.pt",
    )

    # Best checkpoints selected according to
    # policy evaluation reward.
    torch.save(
        best_checkpoint["policy_state_dict"],
        output_dir / "best_policy.pt",
    )

    torch.save(
        best_checkpoint["value_state_dict"],
        output_dir / "best_value.pt",
    )


def main():
    args = parse_args()

    if args.eval_every > args.episodes:
        raise ValueError(
            "--eval-every must be less than or equal "
            "to --episodes."
        )

    run_name = build_run_name(args)

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

    # Initialize the policy exactly as in the other
    # Exercise 2 configurations.
    policy = PolicyNetwork(
        hidden_dim=args.hidden_dim,
    )

    # Save the RNG state immediately after policy
    # initialization. Initializing the ValueNetwork
    # consumes random numbers, but we do not want this
    # extra network to change the subsequent action
    # sampling sequence simply because it exists.
    training_rng_state = (
        torch.random.get_rng_state()
    )

    value_network = ValueNetwork(
        hidden_dim=args.hidden_dim,
    )

    torch.random.set_rng_state(
        training_rng_state,
    )

    policy_optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=args.policy_lr,
    )

    value_optimizer = torch.optim.Adam(
        value_network.parameters(),
        lr=args.value_lr,
    )

    print("Run:", run_name)
    print("Method: value_baseline")
    print("Seed:", args.seed)
    print("Policy learning rate:", args.policy_lr)
    print("Value learning rate:", args.value_lr)
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
        value_losses,
        evaluation_history,
        best_checkpoint,
    ) = train_with_value_baseline(
        env=train_env,
        eval_env=eval_env,
        policy=policy,
        value_network=value_network,
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,
        num_episodes=args.episodes,
        gamma=args.gamma,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
    )

    train_env.close()
    eval_env.close()

    save_results(
        episode_rewards=episode_rewards,
        policy_losses=policy_losses,
        value_losses=value_losses,
        evaluation_history=evaluation_history,
        policy=policy,
        value_network=value_network,
        best_checkpoint=best_checkpoint,
        args=args,
        run_name=run_name,
    )

    print()
    print("Value-baseline training completed")
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