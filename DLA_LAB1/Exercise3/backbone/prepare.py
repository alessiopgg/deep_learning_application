"""Prepare the canonical ResNet-50 GTSRB checkpoint for Exercise 3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from Exercise3.models.gtsrb_transfer import (
    inspect_gtsrb_checkpoint,
    resolve_project_path,
)
from Exercise3.paths import PROJECT_ROOT


DEFAULT_CONFIG = Path("Exercise3/configs/gtsrb_backbone.yaml")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a ResNet-50 classifier on GTSRB with the Exercise 2 "
            "pipeline and publish a canonical checkpoint for Exercise 3."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override experiment.device.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training.epochs.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override data.num_workers.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Retrain even when the canonical checkpoint already exists. "
            "The existing file is replaced only after successful training."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the Exercise 2 command without executing it.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the canonical checkpoint without training.",
    )
    return parser.parse_args()


def _load_config(path: Path) -> DictConfig:
    resolved = resolve_project_path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"GTSRB backbone configuration not found: {resolved}"
        )

    config = OmegaConf.load(resolved)
    OmegaConf.resolve(config)
    _validate_config(config)
    return config


def _validate_config(config: DictConfig) -> None:
    required_sections = {
        "source",
        "output",
        "model",
        "data",
        "training",
        "checkpoint",
        "experiment",
    }
    missing = sorted(required_sections - set(config.keys()))
    if missing:
        raise KeyError(
            "Missing gtsrb_backbone.yaml sections: "
            + ", ".join(missing)
        )

    if str(config.model.name).lower() != "resnet50":
        raise ValueError("The Exercise 3 backbone must be ResNet-50.")
    if str(config.model.classifier_type).lower() != "linear":
        raise ValueError(
            "The canonical Exercise 3 checkpoint must use a linear head."
        )
    if str(config.model.fine_tuning_strategy).lower() != "full":
        raise ValueError(
            "The canonical Exercise 3 checkpoint must use full fine-tuning."
        )
    if int(config.model.num_classes) != 43:
        raise ValueError("GTSRB classification requires 43 classes.")
    if str(config.checkpoint.monitor) != "validation_loss":
        raise ValueError(
            "The canonical checkpoint must be selected by validation_loss."
        )
    if str(config.checkpoint.mode) != "min":
        raise ValueError(
            "validation_loss must be selected with checkpoint.mode=min."
        )
    if int(config.data.batch_size) <= 0:
        raise ValueError("data.batch_size must be positive.")
    if int(config.data.num_workers) < 0:
        raise ValueError("data.num_workers cannot be negative.")
    if int(config.training.epochs) <= 0:
        raise ValueError("training.epochs must be positive.")
    if int(config.experiment.seed) < 0:
        raise ValueError("experiment.seed must be non-negative.")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _resolve_exercise2_output(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / "Exercise2" / path).resolve()


def _new_run_directory(
    output_root: Path,
    before: set[Path],
) -> Path:
    after = {
        path.resolve()
        for path in output_root.iterdir()
        if path.is_dir()
    }
    created = sorted(
        after - before,
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not created:
        raise RuntimeError(
            "Exercise 2 training completed but no new run directory "
            "was created."
        )
    return created[0]


def _exercise2_command(config: DictConfig) -> list[str]:
    return [
        sys.executable,
        str(resolve_project_path(config.source.exercise2_entrypoint)),
        f"paths.data_dir={config.source.exercise2_data_dir}",
        f"paths.output_dir={config.source.exercise2_output_dir}",
        f"model.name={config.model.name}",
        f"model.classifier_type={config.model.classifier_type}",
        (
            "model.fine_tuning_strategy="
            f"{config.model.fine_tuning_strategy}"
        ),
        f"model.num_classes={config.model.num_classes}",
        f"data.validation_size={config.data.validation_size}",
        f"data.batch_size={config.data.batch_size}",
        f"data.num_workers={config.data.num_workers}",
        f"data.pin_memory={str(bool(config.data.pin_memory)).lower()}",
        f"training.epochs={config.training.epochs}",
        (
            "training.backbone_learning_rate="
            f"{config.training.backbone_learning_rate}"
        ),
        (
            "training.classifier_learning_rate="
            f"{config.training.classifier_learning_rate}"
        ),
        f"training.weight_decay={config.training.weight_decay}",
        f"checkpoint.monitor={config.checkpoint.monitor}",
        f"checkpoint.mode={config.checkpoint.mode}",
        f"logging.batch_interval={config.logging.batch_interval}",
        f"experiment.seed={config.experiment.seed}",
        f"experiment.device={config.experiment.device}",
        (
            "experiment.deterministic="
            f"{str(bool(config.experiment.deterministic)).lower()}"
        ),
        f"experiment.run_name={config.experiment.run_name}",
        "experiment.smoke_test_batches=0",
    ]


def _validate_checkpoint(path: Path) -> dict[str, Any]:
    return inspect_gtsrb_checkpoint(
        path,
        required_model="resnet50",
        required_strategy="full",
        required_classifier_type="linear",
    )


def main() -> None:
    arguments = parse_arguments()
    config = _load_config(arguments.config)

    if arguments.device is not None:
        config.experiment.device = arguments.device
    if arguments.epochs is not None:
        config.training.epochs = arguments.epochs
    if arguments.num_workers is not None:
        config.data.num_workers = arguments.num_workers
    _validate_config(config)

    canonical_checkpoint = resolve_project_path(
        config.output.canonical_checkpoint
    ).resolve()
    metadata_path = resolve_project_path(
        config.output.metadata
    ).resolve()
    training_output_root = _resolve_exercise2_output(
        config.source.exercise2_output_dir
    )

    if arguments.validate_only:
        report = _validate_checkpoint(canonical_checkpoint)
        print("\nCanonical GTSRB checkpoint: VALID")
        print(f"Checkpoint: {canonical_checkpoint}")
        print(f"Model: {report['checkpoint_model']}")
        print(f"Strategy: {report['checkpoint_strategy']}")
        print(
            "Classifier: "
            f"{report['checkpoint_classifier_type']}"
        )
        print(
            "Transfer tensors: "
            f"{report['transfer_tensor_count']}"
        )
        return

    if canonical_checkpoint.is_file() and not arguments.force:
        report = _validate_checkpoint(canonical_checkpoint)
        print("\nCanonical GTSRB checkpoint already exists.")
        print(f"Checkpoint: {canonical_checkpoint}")
        print(
            "Use --force only when an intentional retraining is required."
        )
        print(
            "Stored validation loss: "
            f"{report['checkpoint_validation_loss']}"
        )
        return

    training_output_root.mkdir(parents=True, exist_ok=True)
    canonical_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    before = {
        path.resolve()
        for path in training_output_root.iterdir()
        if path.is_dir()
    }
    command = _exercise2_command(config)

    print("\n=== Exercise 3 - Prepare GTSRB backbone ===")
    print("$ " + subprocess.list2cmdline(command), flush=True)

    if arguments.dry_run:
        print("\nDry run completed. No training was started.")
        return

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )

    run_dir = _new_run_directory(training_output_root, before)
    source_checkpoint = run_dir / "best_model.pt"
    if not source_checkpoint.is_file():
        raise FileNotFoundError(
            f"Exercise 2 best checkpoint not found: {source_checkpoint}"
        )

    source_report = _validate_checkpoint(source_checkpoint)

    temporary_checkpoint = canonical_checkpoint.with_name(
        canonical_checkpoint.name + ".tmp"
    )
    shutil.copy2(source_checkpoint, temporary_checkpoint)
    os.replace(temporary_checkpoint, canonical_checkpoint)

    canonical_report = _validate_checkpoint(canonical_checkpoint)
    resolved_config = OmegaConf.to_container(
        config,
        resolve=True,
        enum_to_str=True,
    )
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": (
            "Canonical ResNet-50 GTSRB initialization for Exercise 3 "
            "Faster R-CNN runs B, C and D."
        ),
        "source_pipeline": "Exercise2",
        "source_entrypoint": str(config.source.exercise2_entrypoint),
        "source_run_dir": str(run_dir),
        "source_checkpoint": str(source_checkpoint),
        "canonical_checkpoint": str(canonical_checkpoint),
        "canonical_checkpoint_sha256": _file_sha256(
            canonical_checkpoint
        ),
        "git_commit": _git_commit(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "test_split_used": False,
        "checkpoint_selection": {
            "monitor": str(config.checkpoint.monitor),
            "mode": str(config.checkpoint.mode),
        },
        "source_checkpoint_report": source_report,
        "canonical_checkpoint_report": canonical_report,
        "resolved_prepare_config": resolved_config,
    }
    _atomic_json(metadata_path, metadata)

    print("\nGTSRB backbone preparation completed.")
    print(f"Source run: {run_dir}")
    print(f"Canonical checkpoint: {canonical_checkpoint}")
    print(f"Metadata: {metadata_path}")
    print(
        "Validation loss: "
        f"{canonical_report['checkpoint_validation_loss']}"
    )
    print(
        "Checkpoint SHA-256: "
        f"{metadata['canonical_checkpoint_sha256']}"
    )


if __name__ == "__main__":
    main()
