"""OmegaConf configuration loading and validation for Exercise 2."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from omegaconf import DictConfig, OmegaConf


EXERCISE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = EXERCISE_DIR / "configs" / "default.yaml"

VALID_MODELS = {"resnet18", "resnet50"}
VALID_CLASSIFIERS = {"linear", "mlp"}
VALID_FINE_TUNING_STRATEGIES = {"classifier", "last_block", "full"}
VALID_CHECKPOINT_METRICS = {
    "validation_loss",
    "validation_accuracy",
    "validation_macro_f1",
}
VALID_CHECKPOINT_MODES = {"min", "max"}
VALID_LOSS_FUNCTIONS = {"cross_entropy"}
VALID_OPTIMIZERS = {"adamw"}


@dataclass
class PathsConfig:
    """Paths interpreted relative to the Exercise2 directory."""

    data_dir: str = "../data"
    output_dir: str = "outputs"


@dataclass
class DataConfig:
    """Dataset split and DataLoader settings."""

    dataset_name: str = "GTSRB"
    validation_size: float = 0.20
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = True


@dataclass
class ModelConfig:
    """Backbone, classifier and fine-tuning settings."""

    name: str = "resnet18"
    classifier_type: str = "linear"
    fine_tuning_strategy: str = "last_block"
    num_classes: int = 43
    mlp_hidden_features: int = 256
    mlp_dropout: float = 0.30


@dataclass
class TrainingConfig:
    """Loss, optimizer and training hyperparameters."""

    epochs: int = 5
    loss_function: str = "cross_entropy"
    optimizer: str = "adamw"
    backbone_learning_rate: float = 1e-4
    classifier_learning_rate: float = 1e-3
    weight_decay: float = 1e-4


@dataclass
class CheckpointConfig:
    """Rule used to select the best validation checkpoint."""

    monitor: str = "validation_loss"
    mode: str = "min"


@dataclass
class LoggingConfig:
    """Console and batch-level metric logging settings."""

    batch_interval: int = 50


@dataclass
class TrackingConfig:
    """Optional Weights & Biases tracking settings."""

    use_wandb: bool = False
    project: str = "dla-lab1"
    group: str = "exercise-2"


@dataclass
class ExperimentConfig:
    """General reproducibility and execution settings."""

    seed: int = 42
    device: str = "auto"
    deterministic: bool = False
    run_name: Optional[str] = None
    smoke_test_batches: int = 0


@dataclass
class PipelineConfig:
    """Complete structured configuration for one Exercise 2 run."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


def load_config(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    cli_args: Optional[Sequence[str]] = None,
) -> DictConfig:
    """
    Load the structured defaults, YAML file and command-line overrides.

    Precedence, from lowest to highest:
        dataclass defaults -> YAML file -> command-line overrides
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    structured_config = OmegaConf.structured(PipelineConfig)
    yaml_config = OmegaConf.load(path)
    override_config = OmegaConf.from_cli(
        list(sys.argv[1:] if cli_args is None else cli_args)
    )

    config = OmegaConf.merge(
        structured_config,
        yaml_config,
        override_config,
    )
    OmegaConf.resolve(config)
    validate_config(config)
    return config


def validate_config(config: DictConfig) -> None:
    """Validate relationships and value ranges not covered by type checks."""
    if config.model.name not in VALID_MODELS:
        raise ValueError(
            f"Unknown model '{config.model.name}'. "
            f"Choose one of: {sorted(VALID_MODELS)}"
        )

    if config.model.classifier_type not in VALID_CLASSIFIERS:
        raise ValueError(
            f"Unknown classifier '{config.model.classifier_type}'. "
            f"Choose one of: {sorted(VALID_CLASSIFIERS)}"
        )

    if (
        config.model.fine_tuning_strategy
        not in VALID_FINE_TUNING_STRATEGIES
    ):
        raise ValueError(
            "Unknown fine-tuning strategy "
            f"'{config.model.fine_tuning_strategy}'. Choose one of: "
            f"{sorted(VALID_FINE_TUNING_STRATEGIES)}"
        )

    if config.model.num_classes <= 1:
        raise ValueError("model.num_classes must be greater than one.")

    if config.model.mlp_hidden_features <= 0:
        raise ValueError("model.mlp_hidden_features must be positive.")

    if not 0.0 <= config.model.mlp_dropout < 1.0:
        raise ValueError("model.mlp_dropout must be in [0, 1).")

    if not 0.0 < config.data.validation_size < 1.0:
        raise ValueError("data.validation_size must be between 0 and 1.")

    if config.data.batch_size <= 0:
        raise ValueError("data.batch_size must be positive.")

    if config.data.num_workers < 0:
        raise ValueError("data.num_workers cannot be negative.")

    if config.training.epochs <= 0:
        raise ValueError("training.epochs must be positive.")

    if config.training.loss_function not in VALID_LOSS_FUNCTIONS:
        raise ValueError(
            f"Unknown loss function '{config.training.loss_function}'. "
            f"Choose one of: {sorted(VALID_LOSS_FUNCTIONS)}"
        )

    if config.training.optimizer not in VALID_OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer '{config.training.optimizer}'. "
            f"Choose one of: {sorted(VALID_OPTIMIZERS)}"
        )

    if config.training.backbone_learning_rate <= 0:
        raise ValueError(
            "training.backbone_learning_rate must be positive."
        )

    if config.training.classifier_learning_rate <= 0:
        raise ValueError(
            "training.classifier_learning_rate must be positive."
        )

    if config.training.weight_decay < 0:
        raise ValueError("training.weight_decay cannot be negative.")

    if config.logging.batch_interval <= 0:
        raise ValueError("logging.batch_interval must be positive.")

    if config.checkpoint.monitor not in VALID_CHECKPOINT_METRICS:
        raise ValueError(
            f"Unknown checkpoint metric '{config.checkpoint.monitor}'. "
            f"Choose one of: {sorted(VALID_CHECKPOINT_METRICS)}"
        )

    if config.checkpoint.mode not in VALID_CHECKPOINT_MODES:
        raise ValueError(
            f"Unknown checkpoint mode '{config.checkpoint.mode}'. "
            f"Choose one of: {sorted(VALID_CHECKPOINT_MODES)}"
        )

    if (
        config.checkpoint.monitor == "validation_loss"
        and config.checkpoint.mode != "min"
    ):
        raise ValueError(
            "validation_loss must be monitored with checkpoint.mode=min."
        )

    if (
        config.checkpoint.monitor
        in {"validation_accuracy", "validation_macro_f1"}
        and config.checkpoint.mode != "max"
    ):
        raise ValueError(
            f"{config.checkpoint.monitor} must be monitored with "
            "checkpoint.mode=max."
        )

    if config.experiment.smoke_test_batches < 0:
        raise ValueError(
            "experiment.smoke_test_batches cannot be negative."
        )

    device = config.experiment.device
    if not (
        device in {"auto", "cpu", "cuda"}
        or device.startswith("cuda:")
    ):
        raise ValueError(
            "experiment.device must be auto, cpu, cuda or cuda:<index>."
        )


def config_to_yaml(config: DictConfig) -> str:
    """Return the resolved configuration in YAML format."""
    return OmegaConf.to_yaml(config, resolve=True)
