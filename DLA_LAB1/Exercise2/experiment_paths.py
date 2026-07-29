"""Run-directory creation for Exercise 2 experiments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from omegaconf import DictConfig

from data import resolve_exercise_path


@dataclass(frozen=True)
class RunPaths:
    """Filesystem locations reserved for one experimental run."""

    run_id: str
    run_dir: Path
    checkpoint_path: Path


def _safe_name(value: str) -> str:
    """Convert a user-provided run name to a filesystem-safe token."""
    normalized = "".join(
        character if character.isalnum() or character in {"-", "_"}
        else "-"
        for character in value.strip()
    )
    normalized = normalized.strip("-")
    if not normalized:
        raise ValueError(
            "experiment.run_name must contain at least one letter or digit."
        )
    return normalized


def create_run_paths(config: DictConfig) -> RunPaths:
    """Create a unique run directory and return its main artifact paths."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    configured_name = config.experiment.run_name

    if configured_name is None:
        descriptive_name = (
            f"{config.model.name}-"
            f"{config.model.fine_tuning_strategy}-"
            f"{config.model.classifier_type}"
        )
    else:
        descriptive_name = _safe_name(str(configured_name))

    run_id = f"{timestamp}_{descriptive_name}"
    output_dir = resolve_exercise_path(config.paths.output_dir)
    run_dir = output_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    return RunPaths(
        run_id=run_id,
        run_dir=run_dir,
        checkpoint_path=run_dir / "best_model.pt",
    )
