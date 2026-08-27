"""OmegaConf configuration for Exercise 3 detector training."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from omegaconf import OmegaConf

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
    backbone_source: str = "coco"
    gtsrb_checkpoint: str | None = None
    required_gtsrb_strategy: str | None = "full"
    trainable_backbone: str = "frozen"
    freeze_backbone: bool = True
    num_classes: int = 44
    progress: bool = True


@dataclass
class OptimizerConfig:
    name: str = "sgd"
    learning_rate: float = 0.005
    backbone_learning_rate: float = 0.0001
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
    parser = argparse.ArgumentParser(
        description="Train the Exercise 3.3 Faster R-CNN detector."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parsed, overrides = parser.parse_known_args(arguments)
    invalid = [value for value in overrides if "=" not in value]
    if invalid:
        raise ValueError(
            f"Overrides must use key=value syntax. Invalid: {invalid}"
        )
    return parsed.config, overrides


def load_training_config(
    config_path: Path,
    overrides: Sequence[str] | None = None,
) -> BaselineTrainingConfig:
    path = config_path.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    merged = OmegaConf.merge(
        OmegaConf.structured(BaselineTrainingConfig),
        OmegaConf.load(path),
        OmegaConf.from_dotlist(list(overrides or [])),
    )
    OmegaConf.resolve(merged)
    config = OmegaConf.to_object(merged)
    if not isinstance(config, BaselineTrainingConfig):
        raise TypeError("Invalid structured configuration.")
    validate_training_config(config)
    return config


def validate_training_config(config: BaselineTrainingConfig) -> None:
    d, m, o, s, t, c, w, e = (
        config.data,
        config.model,
        config.optimizer,
        config.scheduler,
        config.training,
        config.checkpoint,
        config.tracking,
        config.experiment,
    )

    if d.train_batch_size <= 0 or d.evaluation_batch_size <= 0:
        raise ValueError("data batch sizes must be > 0.")
    if d.num_workers < 0:
        raise ValueError("data.num_workers cannot be negative.")
    if d.persistent_workers and d.num_workers == 0:
        raise ValueError("persistent_workers requires num_workers > 0.")

    if m.architecture != "fasterrcnn_resnet50_fpn":
        raise ValueError("Only fasterrcnn_resnet50_fpn is supported.")
    if m.weights not in {"coco", "none"}:
        raise ValueError("model.weights must be coco or none.")
    if m.backbone_source not in {"coco", "gtsrb"}:
        raise ValueError("model.backbone_source must be coco or gtsrb.")
    if m.trainable_backbone not in {"frozen", "layer4", "layer3_layer4"}:
        raise ValueError("Invalid model.trainable_backbone.")
    if m.freeze_backbone != (m.trainable_backbone == "frozen"):
        raise ValueError("freeze_backbone conflicts with trainable_backbone.")
    if m.backbone_source == "gtsrb":
        if not m.gtsrb_checkpoint:
            raise ValueError("GTSRB backbone requires model.gtsrb_checkpoint.")
        if m.weights != "coco":
            raise ValueError("GTSRB runs must retain COCO detector weights.")
    if m.num_classes != 44:
        raise ValueError("The verified detector taxonomy requires 44 classes.")

    if o.name != "sgd":
        raise ValueError("optimizer.name must be sgd.")
    if o.learning_rate <= 0 or o.backbone_learning_rate <= 0:
        raise ValueError("Optimizer learning rates must be > 0.")
    if o.backbone_learning_rate > o.learning_rate:
        raise ValueError("Backbone LR cannot exceed detector LR.")
    if not 0 <= o.momentum < 1 or o.weight_decay < 0:
        raise ValueError("Invalid optimizer momentum or weight decay.")

    if s.name != "step_lr" or s.step_size <= 0 or not 0 < s.gamma <= 1:
        raise ValueError("Invalid StepLR configuration.")

    if t.epochs <= 0 or t.amp_initial_scale <= 0 or t.logging_interval <= 0:
        raise ValueError("Invalid training epochs/AMP scale/logging interval.")
    if t.gradient_clip_norm is not None and t.gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be null or > 0.")
    for name in ("max_train_batches", "max_validation_batches"):
        value = getattr(t, name)
        if value is not None and value <= 0:
            raise ValueError(f"training.{name} must be null or > 0.")

    if c.monitor != "validation_total_loss" or c.mode != "min":
        raise ValueError(
            "Checkpoint selection must minimize validation_total_loss."
        )
    if w.mode not in {"online", "offline", "disabled"}:
        raise ValueError("tracking.mode must be online, offline or disabled.")
    if e.seed < 0:
        raise ValueError("experiment.seed must be non-negative.")


def resolve_exercise_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else EXERCISE_DIR / path


def resolve_cache_dir(config: BaselineTrainingConfig) -> Path | None:
    return (
        None
        if config.paths.cache_dir is None
        else resolve_exercise_path(config.paths.cache_dir)
    )


def resolve_output_root(config: BaselineTrainingConfig) -> Path:
    path = resolve_exercise_path(config.paths.output_dir)
    if path == OUTPUT_DIR:
        raise ValueError("paths.output_dir must be below outputs/.")
    return path


def save_resolved_config(
    config: BaselineTrainingConfig,
    run_dir: Path,
) -> tuple[Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    data = config.to_dict()
    yaml_path, json_path = run_dir / "config.yaml", run_dir / "config.json"
    yaml_path.write_text(
        OmegaConf.to_yaml(OmegaConf.create(data), resolve=True),
        encoding="utf-8",
    )
    json_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return yaml_path, json_path


def validate_resume_compatibility(
    current: BaselineTrainingConfig,
    checkpoint_config: dict[str, Any],
) -> None:
    current_config = current.to_dict()
    for section in ("data", "model", "optimizer", "scheduler"):
        if checkpoint_config.get(section) != current_config.get(section):
            raise ValueError(f"Resume mismatch in section {section!r}.")

    for key in (
        "amp",
        "amp_initial_scale",
        "gradient_clip_norm",
        "max_train_batches",
        "max_validation_batches",
    ):
        if checkpoint_config.get("training", {}).get(key) != getattr(
            current.training, key
        ):
            raise ValueError(f"Resume mismatch for training.{key}.")

    for key in ("seed", "deterministic"):
        if checkpoint_config.get("experiment", {}).get(key) != getattr(
            current.experiment, key
        ):
            raise ValueError(f"Resume mismatch for experiment.{key}.")
