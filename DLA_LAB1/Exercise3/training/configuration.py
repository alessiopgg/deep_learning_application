"""OmegaConf configuration for the Exercise 3.3 Faster R-CNN baseline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from omegaconf import DictConfig, OmegaConf

from Exercise3.paths import EXERCISE_DIR, OUTPUT_DIR


DEFAULT_CONFIG_PATH = EXERCISE_DIR / "configs" / "baseline.yaml"


@dataclass
class PathConfig:
    cache_dir: str | None = None
    output_dir: str = "outputs/step_11/runs"


@dataclass
class DataConfig:
    train_batch_size: int = 1
    evaluation_batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False


@dataclass
class ModelConfig:
    architecture: str = "fasterrcnn_resnet50_fpn"
    weights: str = "coco"
    freeze_backbone: bool = True
    num_classes: int = 44
    progress: bool = True


@dataclass
class OptimizerConfig:
    name: str = "sgd"
    learning_rate: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0005


@dataclass
class SchedulerConfig:
    name: str = "step_lr"
    step_size: int = 3
    gamma: float = 0.1


@dataclass
class TrainingConfig:
    epochs: int = 5
    amp: bool = True
    amp_initial_scale: float = 1024.0
    gradient_clip_norm: float | None = None
    logging_interval: int = 25
    max_train_batches: int | None = None
    max_validation_batches: int | None = None


@dataclass
class CheckpointConfig:
    monitor: str = "validation_total_loss"
    mode: str = "min"
    save_last: bool = True


@dataclass
class TrackingConfig:
    use_wandb: bool = False
    project: str = "dla-lab1"
    entity: str | None = None
    group: str = "exercise-3-3-baseline"
    mode: str = "online"
    log_best_checkpoint: bool = False


@dataclass
class ExperimentConfig:
    seed: int = 42
    device: str = "auto"
    deterministic: bool = False
    run_name: str | None = None
    resume_from: str | None = None


@dataclass
class BaselineTrainingConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_config_arguments(
    arguments: Sequence[str] | None = None,
) -> tuple[Path, list[str]]:
    """Parse only --config; every remaining token is an OmegaConf override."""
    parser = argparse.ArgumentParser(
        description="Train the Exercise 3.3 Faster R-CNN baseline.",
        add_help=True,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="YAML configuration file.",
    )
    parsed, overrides = parser.parse_known_args(arguments)
    invalid = [item for item in overrides if "=" not in item]
    if invalid:
        raise ValueError(
            "Unknown arguments must be OmegaConf dot-list overrides of the "
            f"form key=value. Invalid: {invalid}."
        )
    return parsed.config, overrides


def load_training_config(
    config_path: Path,
    overrides: Sequence[str] | None = None,
) -> BaselineTrainingConfig:
    """Load defaults, YAML and command-line overrides in increasing priority."""
    resolved_path = config_path.expanduser()
    if not resolved_path.is_absolute():
        resolved_path = Path.cwd() / resolved_path
    if not resolved_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {resolved_path}")

    schema = OmegaConf.structured(BaselineTrainingConfig)
    yaml_config = OmegaConf.load(resolved_path)
    override_config = OmegaConf.from_dotlist(list(overrides or []))
    merged: DictConfig = OmegaConf.merge(schema, yaml_config, override_config)
    OmegaConf.resolve(merged)

    config = OmegaConf.to_object(merged)
    if not isinstance(config, BaselineTrainingConfig):
        raise TypeError("OmegaConf did not produce BaselineTrainingConfig.")
    validate_training_config(config)
    return config


def validate_training_config(config: BaselineTrainingConfig) -> None:
    if config.data.train_batch_size <= 0:
        raise ValueError("data.train_batch_size must be greater than zero.")
    if config.data.evaluation_batch_size <= 0:
        raise ValueError("data.evaluation_batch_size must be greater than zero.")
    if config.data.num_workers < 0:
        raise ValueError("data.num_workers cannot be negative.")
    if config.data.persistent_workers and config.data.num_workers == 0:
        raise ValueError(
            "data.persistent_workers=true requires data.num_workers > 0."
        )

    if config.model.architecture != "fasterrcnn_resnet50_fpn":
        raise ValueError(
            "The baseline supports only model.architecture="
            "fasterrcnn_resnet50_fpn."
        )
    if config.model.weights not in {"coco", "none"}:
        raise ValueError("model.weights must be coco or none.")
    if not config.model.freeze_backbone:
        raise ValueError(
            "Step 11 is the frozen-backbone baseline; "
            "model.freeze_backbone must remain true."
        )
    if config.model.num_classes != 44:
        raise ValueError("The verified detector taxonomy requires 44 classes.")

    if config.optimizer.name != "sgd":
        raise ValueError("The Step 11 baseline supports optimizer.name=sgd.")
    if config.optimizer.learning_rate <= 0:
        raise ValueError("optimizer.learning_rate must be greater than zero.")
    if not 0 <= config.optimizer.momentum < 1:
        raise ValueError("optimizer.momentum must satisfy 0 <= momentum < 1.")
    if config.optimizer.weight_decay < 0:
        raise ValueError("optimizer.weight_decay cannot be negative.")

    if config.scheduler.name != "step_lr":
        raise ValueError("The Step 11 baseline supports scheduler.name=step_lr.")
    if config.scheduler.step_size <= 0:
        raise ValueError("scheduler.step_size must be greater than zero.")
    if not 0 < config.scheduler.gamma <= 1:
        raise ValueError("scheduler.gamma must satisfy 0 < gamma <= 1.")

    if config.training.epochs <= 0:
        raise ValueError("training.epochs must be greater than zero.")
    if config.training.amp_initial_scale <= 0:
        raise ValueError("training.amp_initial_scale must be greater than zero.")
    if (
        config.training.gradient_clip_norm is not None
        and config.training.gradient_clip_norm <= 0
    ):
        raise ValueError(
            "training.gradient_clip_norm must be null or greater than zero."
        )
    if config.training.logging_interval <= 0:
        raise ValueError("training.logging_interval must be greater than zero.")
    for name, value in (
        ("max_train_batches", config.training.max_train_batches),
        ("max_validation_batches", config.training.max_validation_batches),
    ):
        if value is not None and value <= 0:
            raise ValueError(f"training.{name} must be null or greater than zero.")

    if config.checkpoint.monitor != "validation_total_loss":
        raise ValueError(
            "Step 11 currently monitors checkpoint.validation_total_loss only."
        )
    if config.checkpoint.mode != "min":
        raise ValueError("validation_total_loss requires checkpoint.mode=min.")

    if config.tracking.mode not in {"online", "offline", "disabled"}:
        raise ValueError(
            "tracking.mode must be online, offline or disabled."
        )
    if config.experiment.seed < 0:
        raise ValueError("experiment.seed must be non-negative.")


def resolve_exercise_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else EXERCISE_DIR / path


def resolve_cache_dir(config: BaselineTrainingConfig) -> Path | None:
    if config.paths.cache_dir is None:
        return None
    return resolve_exercise_path(config.paths.cache_dir)


def resolve_output_root(config: BaselineTrainingConfig) -> Path:
    path = resolve_exercise_path(config.paths.output_dir)
    # Protect against accidentally resolving to the whole Exercise3 output root
    # when the configuration value is empty.
    if path == OUTPUT_DIR:
        raise ValueError(
            "paths.output_dir must identify a run collection below outputs/."
        )
    return path


def save_resolved_config(
    config: BaselineTrainingConfig,
    run_dir: Path,
) -> tuple[Path, Path]:
    """Save the exact resolved configuration in YAML and JSON formats."""
    run_dir.mkdir(parents=True, exist_ok=True)
    config_dict = config.to_dict()
    yaml_path = run_dir / "config.yaml"
    json_path = run_dir / "config.json"

    yaml_path.write_text(
        OmegaConf.to_yaml(OmegaConf.create(config_dict), resolve=True),
        encoding="utf-8",
    )
    json_path.write_text(
        json.dumps(config_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return yaml_path, json_path


def validate_resume_compatibility(
    current: BaselineTrainingConfig,
    checkpoint_config: dict[str, Any],
) -> None:
    """Reject resume attempts that would silently change the experiment."""
    current_dict = current.to_dict()
    required_paths = (
        "data",
        "model",
        "optimizer",
        "scheduler",
    )
    for key in required_paths:
        if checkpoint_config.get(key) != current_dict.get(key):
            raise ValueError(
                f"Resume configuration mismatch in section {key!r}. "
                "Use the same settings as the original run."
            )

    checkpoint_training = checkpoint_config.get("training", {})
    current_training = current_dict["training"]
    for key in (
        "amp",
        "amp_initial_scale",
        "gradient_clip_norm",
        "max_train_batches",
        "max_validation_batches",
    ):
        if checkpoint_training.get(key) != current_training.get(key):
            raise ValueError(
                f"Resume configuration mismatch for training.{key}."
            )

    checkpoint_experiment = checkpoint_config.get("experiment", {})
    for key in ("seed", "deterministic"):
        if checkpoint_experiment.get(key) != current_dict["experiment"].get(key):
            raise ValueError(
                f"Resume configuration mismatch for experiment.{key}."
            )
