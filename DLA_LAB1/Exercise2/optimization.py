"""Loss and optimizer factories for the Exercise 2 pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn
from torchvision.models.resnet import ResNet


@dataclass(frozen=True)
class OptimizationInfo:
    """Serializable description of the loss and optimizer configuration."""

    loss_function: str
    optimizer: str
    backbone_learning_rate: float
    classifier_learning_rate: float
    weight_decay: float
    parameter_groups: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        result = asdict(self)
        result["parameter_groups"] = list(self.parameter_groups)
        return result


@dataclass(frozen=True)
class TrainingComponents:
    """Loss, optimizer and their descriptive configuration."""

    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    info: OptimizationInfo


def create_loss(config: DictConfig) -> nn.Module:
    """Create the configured classification loss."""
    if config.training.loss_function == "cross_entropy":
        return nn.CrossEntropyLoss()

    raise ValueError(
        f"Unknown loss function '{config.training.loss_function}'. "
        "Choose one of: ['cross_entropy']"
    )


def _create_parameter_groups(
    model: ResNet,
    strategy: str,
    backbone_learning_rate: float,
    classifier_learning_rate: float,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    """Create non-overlapping optimizer groups for one fine-tuning policy."""
    if strategy == "classifier":
        return (
            [
                {
                    "params": model.fc.parameters(),
                    "lr": classifier_learning_rate,
                }
            ],
            ("classifier",),
        )

    if strategy == "last_block":
        return (
            [
                {
                    "params": model.layer4.parameters(),
                    "lr": backbone_learning_rate,
                },
                {
                    "params": model.fc.parameters(),
                    "lr": classifier_learning_rate,
                },
            ],
            ("last_block", "classifier"),
        )

    if strategy == "full":
        backbone_parameters = [
            parameter
            for parameter_name, parameter in model.named_parameters()
            if not parameter_name.startswith("fc.")
        ]
        return (
            [
                {
                    "params": backbone_parameters,
                    "lr": backbone_learning_rate,
                },
                {
                    "params": model.fc.parameters(),
                    "lr": classifier_learning_rate,
                },
            ],
            ("backbone", "classifier"),
        )

    raise ValueError(
        f"Unknown fine-tuning strategy '{strategy}'. "
        "Choose one of: ['classifier', 'last_block', 'full']"
    )


def create_optimizer(
    model: ResNet,
    config: DictConfig,
) -> tuple[torch.optim.Optimizer, tuple[str, ...]]:
    """Create the configured optimizer and its named parameter groups."""
    parameter_groups, group_names = _create_parameter_groups(
        model=model,
        strategy=config.model.fine_tuning_strategy,
        backbone_learning_rate=config.training.backbone_learning_rate,
        classifier_learning_rate=config.training.classifier_learning_rate,
    )

    if config.training.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            parameter_groups,
            weight_decay=config.training.weight_decay,
        )
        return optimizer, group_names

    raise ValueError(
        f"Unknown optimizer '{config.training.optimizer}'. "
        "Choose one of: ['adamw']"
    )


def create_training_components(
    model: ResNet,
    config: DictConfig,
) -> TrainingComponents:
    """Create loss and optimizer without coupling them to the training loop."""
    criterion = create_loss(config)
    optimizer, parameter_groups = create_optimizer(model, config)

    info = OptimizationInfo(
        loss_function=config.training.loss_function,
        optimizer=config.training.optimizer,
        backbone_learning_rate=float(
            config.training.backbone_learning_rate
        ),
        classifier_learning_rate=float(
            config.training.classifier_learning_rate
        ),
        weight_decay=float(config.training.weight_decay),
        parameter_groups=parameter_groups,
    )
    return TrainingComponents(
        criterion=criterion,
        optimizer=optimizer,
        info=info,
    )


def print_optimization_summary(
    components: TrainingComponents,
) -> None:
    """Print loss, optimizer and parameter-group information."""
    info = components.info

    print("\n=== Exercise 2 optimization configuration ===")
    print(f"Loss function: {info.loss_function}")
    print(f"Optimizer: {info.optimizer}")
    print(f"Backbone learning rate: {info.backbone_learning_rate}")
    print(f"Classifier learning rate: {info.classifier_learning_rate}")
    print(f"Weight decay: {info.weight_decay}")
    print(f"Parameter groups: {list(info.parameter_groups)}")
