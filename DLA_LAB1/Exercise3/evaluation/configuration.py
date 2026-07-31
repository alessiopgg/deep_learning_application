"""Structured configuration for Exercise 3.3 detector evaluation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from omegaconf import DictConfig, OmegaConf

from Exercise3.paths import EXERCISE_DIR


DEFAULT_CONFIG_PATH = EXERCISE_DIR / "configs" / "evaluation.yaml"


@dataclass
class CheckpointConfig:
    path: str | None = None
    strict: bool = True


@dataclass
class RuntimeConfig:
    device: str = "auto"
    amp: bool = True
    batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False
    seed: int = 42


@dataclass
class EvaluationConfig:
    split: str = "validation"
    allow_test: bool = False
    backend: str = "auto"
    model_score_threshold: float = 0.0
    fixed_score_threshold: float = 0.5
    fixed_iou_threshold: float = 0.5
    class_metrics: bool = True
    max_batches: int | None = None
    save_predictions: bool = True


@dataclass
class VisualizationConfig:
    enabled: bool = True
    samples: int = 5
    score_threshold: float = 0.25
    box_width: int = 4
    font_path: str | None = None
    font_size: int = 20


@dataclass
class OutputConfig:
    directory: str | None = None
    save_training_plots: bool = True
    save_markdown_summary: bool = True


@dataclass
class TrackingConfig:
    enabled: bool = False
    resume_training_run: bool = True
    log_per_class_table: bool = True
    log_images: bool = True
    log_evaluation_artifact: bool = False


@dataclass
class DetectorEvaluationConfig:
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_evaluation_arguments(
    arguments: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Exercise 3.3 Faster R-CNN checkpoint."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=("validation", "test"), default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Required acknowledgement before evaluating the test split.",
    )
    parsed, overrides = parser.parse_known_args(arguments)
    invalid = [item for item in overrides if "=" not in item]
    if invalid:
        raise ValueError(
            "Unknown arguments must be OmegaConf dot-list overrides. "
            f"Invalid: {invalid}."
        )
    return parsed, overrides


def load_evaluation_config(
    config_path: Path,
    overrides: Sequence[str] | None = None,
) -> DetectorEvaluationConfig:
    path = config_path.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation configuration not found: {path}")

    schema = OmegaConf.structured(DetectorEvaluationConfig)
    yaml_config = OmegaConf.load(path)
    override_config = OmegaConf.from_dotlist(list(overrides or []))
    merged: DictConfig = OmegaConf.merge(schema, yaml_config, override_config)
    OmegaConf.resolve(merged)
    config = OmegaConf.to_object(merged)
    if not isinstance(config, DetectorEvaluationConfig):
        raise TypeError("OmegaConf did not produce DetectorEvaluationConfig.")
    validate_evaluation_config(config, require_checkpoint=False)
    return config


def apply_explicit_arguments(
    config: DetectorEvaluationConfig,
    arguments: argparse.Namespace,
) -> DetectorEvaluationConfig:
    if arguments.checkpoint is not None:
        config.checkpoint.path = str(arguments.checkpoint.expanduser().resolve())
    if arguments.split is not None:
        config.evaluation.split = arguments.split
    if arguments.device is not None:
        config.runtime.device = arguments.device
    if arguments.output_dir is not None:
        config.output.directory = str(arguments.output_dir.expanduser().resolve())
    if arguments.allow_test:
        config.evaluation.allow_test = True
    validate_evaluation_config(config, require_checkpoint=True)
    return config


def validate_evaluation_config(
    config: DetectorEvaluationConfig,
    *,
    require_checkpoint: bool,
) -> None:
    if require_checkpoint and not config.checkpoint.path:
        raise ValueError("A checkpoint is required. Use --checkpoint PATH.")
    if config.runtime.batch_size <= 0:
        raise ValueError("runtime.batch_size must be greater than zero.")
    if config.runtime.num_workers < 0:
        raise ValueError("runtime.num_workers cannot be negative.")
    if config.runtime.persistent_workers and config.runtime.num_workers == 0:
        raise ValueError(
            "runtime.persistent_workers=true requires runtime.num_workers > 0."
        )
    if config.runtime.seed < 0:
        raise ValueError("runtime.seed must be non-negative.")

    if config.evaluation.split not in {"validation", "test"}:
        raise ValueError("evaluation.split must be validation or test.")
    if config.evaluation.split == "test" and not config.evaluation.allow_test:
        raise ValueError(
            "Test evaluation is protected. Re-run with --allow-test only after "
            "the model-selection protocol is complete."
        )
    if config.evaluation.backend not in {
        "auto",
        "pycocotools",
        "faster_coco_eval",
    }:
        raise ValueError(
            "evaluation.backend must be auto, pycocotools or faster_coco_eval."
        )
    for name, value in (
        ("model_score_threshold", config.evaluation.model_score_threshold),
        ("fixed_score_threshold", config.evaluation.fixed_score_threshold),
        ("fixed_iou_threshold", config.evaluation.fixed_iou_threshold),
        ("visualization.score_threshold", config.visualization.score_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be within [0,1].")
    if config.evaluation.max_batches is not None and config.evaluation.max_batches <= 0:
        raise ValueError("evaluation.max_batches must be null or positive.")
    if config.visualization.samples <= 0:
        raise ValueError("visualization.samples must be greater than zero.")
    if config.visualization.box_width <= 0:
        raise ValueError("visualization.box_width must be greater than zero.")
    if config.visualization.font_size <= 0:
        raise ValueError("visualization.font_size must be greater than zero.")


def resolve_checkpoint_path(config: DetectorEvaluationConfig) -> Path:
    if not config.checkpoint.path:
        raise ValueError("checkpoint.path is missing.")
    path = Path(config.checkpoint.path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path.resolve()


def resolve_output_directory(
    config: DetectorEvaluationConfig,
    checkpoint_path: Path,
) -> Path:
    if config.output.directory is None:
        return checkpoint_path.parent / "evaluation" / config.evaluation.split
    path = Path(config.output.directory).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()
