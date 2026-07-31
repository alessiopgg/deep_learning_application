"""Configuration model for the sequential detector experiment matrix."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from Exercise3.paths import PROJECT_ROOT


@dataclass(frozen=True)
class ExperimentSpec:
    id: str
    name: str
    enabled: bool
    backbone_source: str
    trainable_backbone: str
    detector_learning_rate: float
    backbone_learning_rate: float
    extra_training_overrides: tuple[str, ...] = ()
    extra_evaluation_overrides: tuple[str, ...] = ()


@dataclass
class StudyConfig:
    name: str = "exercise-3-3-backbone-study"
    output_dir: str = "Exercise3/outputs/step_18/studies"
    training_config: str = "Exercise3/configs/baseline.yaml"
    evaluation_config: str = "Exercise3/configs/evaluation.yaml"
    device: str = "cuda:0"
    num_workers: int = 4
    persistent_workers: bool = True
    continue_on_error: bool = False
    required_gtsrb_strategy: str | None = "full"
    gtsrb_checkpoint: str | None = None
    common_training_overrides: list[str] = field(default_factory=list)
    common_evaluation_overrides: list[str] = field(default_factory=list)
    wandb_enabled: bool = True
    wandb_project: str = "dla-lab1"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_mode: str = "online"
    log_best_checkpoints: bool = False
    log_evaluation_artifacts: bool = False
    experiments: list[ExperimentSpec] = field(default_factory=list)

    def enabled_experiments(self) -> list[ExperimentSpec]:
        return [experiment for experiment in self.experiments if experiment.enabled]


def _as_plain_dict(path: Path) -> dict[str, Any]:
    loaded = OmegaConf.load(path)
    plain = OmegaConf.to_container(loaded, resolve=True)
    if not isinstance(plain, dict):
        raise TypeError("Experiment matrix YAML root must be a mapping.")
    return plain


def load_study_config(path: Path) -> StudyConfig:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    if not resolved.is_file():
        raise FileNotFoundError(f"Experiment matrix not found: {resolved}")
    root = _as_plain_dict(resolved)
    study = root.get("study", root)
    if not isinstance(study, dict):
        raise TypeError("The study section must be a mapping.")

    raw_experiments = study.get("experiments", [])
    if not isinstance(raw_experiments, list):
        raise TypeError("study.experiments must be a list.")
    experiments = []
    for raw in raw_experiments:
        if not isinstance(raw, dict):
            raise TypeError("Every experiment entry must be a mapping.")
        experiments.append(
            ExperimentSpec(
                id=str(raw["id"]),
                name=str(raw["name"]),
                enabled=bool(raw.get("enabled", True)),
                backbone_source=str(raw["backbone_source"]),
                trainable_backbone=str(raw["trainable_backbone"]),
                detector_learning_rate=float(raw["detector_learning_rate"]),
                backbone_learning_rate=float(raw["backbone_learning_rate"]),
                extra_training_overrides=tuple(raw.get("extra_training_overrides", [])),
                extra_evaluation_overrides=tuple(raw.get("extra_evaluation_overrides", [])),
            )
        )

    config = StudyConfig(
        name=str(study.get("name", "exercise-3-3-backbone-study")),
        output_dir=str(study.get("output_dir", "Exercise3/outputs/step_18/studies")),
        training_config=str(study.get("training_config", "Exercise3/configs/baseline.yaml")),
        evaluation_config=str(study.get("evaluation_config", "Exercise3/configs/evaluation.yaml")),
        device=str(study.get("device", "cuda:0")),
        num_workers=int(study.get("num_workers", 4)),
        persistent_workers=bool(study.get("persistent_workers", True)),
        continue_on_error=bool(study.get("continue_on_error", False)),
        required_gtsrb_strategy=study.get("required_gtsrb_strategy", "full"),
        gtsrb_checkpoint=study.get("gtsrb_checkpoint"),
        common_training_overrides=list(study.get("common_training_overrides", [])),
        common_evaluation_overrides=list(study.get("common_evaluation_overrides", [])),
        wandb_enabled=bool(study.get("wandb_enabled", True)),
        wandb_project=str(study.get("wandb_project", "dla-lab1")),
        wandb_entity=study.get("wandb_entity"),
        wandb_group=study.get("wandb_group"),
        wandb_mode=str(study.get("wandb_mode", "online")),
        log_best_checkpoints=bool(study.get("log_best_checkpoints", False)),
        log_evaluation_artifacts=bool(study.get("log_evaluation_artifacts", False)),
        experiments=experiments,
    )
    validate_study_config(config)
    return config


def validate_study_config(config: StudyConfig) -> None:
    if not config.name.strip():
        raise ValueError("study.name cannot be empty.")
    if config.num_workers < 0:
        raise ValueError("study.num_workers cannot be negative.")
    if config.persistent_workers and config.num_workers == 0:
        raise ValueError("persistent_workers=true requires num_workers > 0.")
    if config.wandb_mode not in {"online", "offline", "disabled"}:
        raise ValueError("wandb_mode must be online, offline or disabled.")
    if not config.experiments:
        raise ValueError("The matrix contains no experiments.")

    ids = [experiment.id for experiment in config.experiments]
    if len(ids) != len(set(ids)):
        raise ValueError("Experiment IDs must be unique.")
    for experiment in config.experiments:
        if experiment.backbone_source not in {"coco", "gtsrb"}:
            raise ValueError(
                f"Experiment {experiment.id}: invalid backbone_source."
            )
        if experiment.trainable_backbone not in {
            "frozen",
            "layer4",
            "layer3_layer4",
        }:
            raise ValueError(
                f"Experiment {experiment.id}: invalid trainable_backbone."
            )
        if experiment.detector_learning_rate <= 0:
            raise ValueError(
                f"Experiment {experiment.id}: detector LR must be positive."
            )
        if experiment.backbone_learning_rate <= 0:
            raise ValueError(
                f"Experiment {experiment.id}: backbone LR must be positive."
            )
        if experiment.backbone_learning_rate > experiment.detector_learning_rate:
            raise ValueError(
                f"Experiment {experiment.id}: backbone LR exceeds detector LR."
            )


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path
