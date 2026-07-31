"""Unified command-line entry point for Exercise 3.

This module provides one public CLI while keeping the implementation
of dataset inspection, analysis, training, evaluation and experiment
orchestration in their dedicated modules.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CommandSpec:
    """Describe a command delegated to another Exercise3 module."""

    module: str
    description: str


COMMANDS: dict[str, CommandSpec] = {
    "inspect": CommandSpec(
        module="Exercise3.inspect_dataset",
        description="Load and inspect the detection dataset.",
    ),
    "eda": CommandSpec(
        module="Exercise3.analysis.eda",
        description="Run exploratory data analysis.",
    ),
    "class-mapping": CommandSpec(
        module="Exercise3.analysis.class_mapping",
        description="Validate the detection-to-GTSRB class mapping.",
    ),
    "train": CommandSpec(
        module="Exercise3.train_baseline",
        description="Train one Faster R-CNN detector configuration.",
    ),
    "evaluate": CommandSpec(
        module="Exercise3.evaluate_detector",
        description="Evaluate a detector checkpoint.",
    ),
    "matrix": CommandSpec(
        module="Exercise3.run_experiment_matrix",
        description="Run or resume the A-D experiment matrix.",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser used only to select a command."""

    command_lines = "\n".join(
        f"  {name:<15} {spec.description}"
        for name, spec in COMMANDS.items()
    )

    return argparse.ArgumentParser(
        prog="python -m Exercise3.main",
        description=(
            "Unified entry point for Exercise 3. Arguments written after "
            "the command are forwarded unchanged to the selected module."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available commands:\n"
            f"{command_lines}\n\n"
            "Examples:\n"
            "  python -m Exercise3.main inspect --split train\n"
            "  python -m Exercise3.main eda\n"
            "  python -m Exercise3.main train "
            "--config Exercise3/configs/baseline.yaml\n"
            "  python -m Exercise3.main evaluate --help\n"
            "  python -m Exercise3.main matrix --preflight-only --no-wandb"
        ),
    )


def run_module(module: str, arguments: Sequence[str]) -> int:
    """Execute an Exercise3 module in an isolated Python subprocess."""

    command = [
        sys.executable,
        "-m",
        module,
        *arguments,
    ]

    print("$ " + " ".join(command), flush=True)
    completed_process = subprocess.run(
        command,
        check=False,
    )
    return int(completed_process.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    """Select a command and forward the remaining arguments."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    if not arguments or arguments[0] in {"-h", "--help"}:
        parser.print_help()
        return 0

    command_name = arguments[0]
    command_spec = COMMANDS.get(command_name)

    if command_spec is None:
        available = ", ".join(COMMANDS)
        parser.error(
            f"unknown command {command_name!r}. "
            f"Available commands: {available}"
        )

    return run_module(
        command_spec.module,
        arguments[1:],
    )


if __name__ == "__main__":
    raise SystemExit(main())
