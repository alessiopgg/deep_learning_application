"""Unified CLI for Exercise 3."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence

COMMANDS = {
    "inspect": ("Exercise3.inspect_dataset", "Inspect the detection dataset."),
    "eda": ("Exercise3.analysis.eda", "Run exploratory data analysis."),
    "class-mapping": (
        "Exercise3.analysis.class_mapping",
        "Validate the detection-to-GTSRB class mapping.",
    ),
    "prepare-backbone": (
        "Exercise3.backbone.prepare",
        "Train and publish the canonical GTSRB ResNet-50.",
    ),
    "train": ("Exercise3.train_baseline", "Train one Faster R-CNN configuration."),
    "evaluate": ("Exercise3.evaluate_detector", "Evaluate a detector checkpoint."),
    "matrix": (
        "Exercise3.run_experiment_matrix",
        "Run or resume the A-D experiment matrix.",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    commands = "\n".join(
        f"  {name:<18} {description}"
        for name, (_, description) in COMMANDS.items()
    )
    return argparse.ArgumentParser(
        prog="python -m Exercise3.main",
        description=(
            "Unified Exercise 3 entry point. Arguments after the command are "
            "forwarded unchanged to the selected module."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available commands:\n{commands}",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not args or args[0] in {"-h", "--help"}:
        parser.print_help()
        return 0

    command = COMMANDS.get(args[0])
    if command is None:
        parser.error(
            f"unknown command {args[0]!r}. Available: {', '.join(COMMANDS)}"
        )

    module, _ = command
    process = subprocess.run(
        [sys.executable, "-m", module, *args[1:]],
        check=False,
    )
    return int(process.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
