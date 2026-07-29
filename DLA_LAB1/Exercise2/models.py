"""Model construction and fine-tuning policies for Exercise 2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import torch
from omegaconf import DictConfig
from torch import nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    resnet18,
    resnet50,
)
from torchvision.models.resnet import ResNet


@dataclass(frozen=True)
class BackboneSpec:
    """Constructor and pretrained weights associated with one backbone."""

    constructor: Callable[..., ResNet]
    weights: Any


@dataclass(frozen=True)
class ModelInfo:
    """Serializable summary of the configured fine-tuning model."""

    model_name: str
    pretrained_weights: str
    classifier_type: str
    fine_tuning_strategy: str
    classifier_input_features: int
    classifier_output_classes: int
    total_parameters: int
    trainable_parameters: int
    trainable_percentage: float
    trainable_modules: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert the summary to a JSON-compatible dictionary."""
        result = asdict(self)
        result["trainable_modules"] = list(self.trainable_modules)
        return result


@dataclass(frozen=True)
class ModelBundle:
    """Configured model together with its descriptive information."""

    model: ResNet
    info: ModelInfo


BACKBONE_REGISTRY: dict[str, BackboneSpec] = {
    "resnet18": BackboneSpec(
        constructor=resnet18,
        weights=ResNet18_Weights.IMAGENET1K_V1,
    ),
    "resnet50": BackboneSpec(
        constructor=resnet50,
        weights=ResNet50_Weights.IMAGENET1K_V2,
    ),
}


def get_backbone_spec(model_name: str) -> BackboneSpec:
    """Return the registered specification for a supported backbone."""
    try:
        return BACKBONE_REGISTRY[model_name]
    except KeyError as error:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose one of: {sorted(BACKBONE_REGISTRY)}"
        ) from error


def create_input_transform(model_name: str):
    """Create the preprocessing required by the selected pretrained weights."""
    return get_backbone_spec(model_name).weights.transforms()


def create_classifier(
    input_features: int,
    num_classes: int,
    classifier_type: str,
    hidden_features: int,
    dropout: float,
) -> nn.Module:
    """Build the linear or MLP classification head requested by the config."""
    if classifier_type == "linear":
        return nn.Linear(input_features, num_classes)

    if classifier_type == "mlp":
        return nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_features, num_classes),
        )

    raise ValueError(
        f"Unknown classifier type '{classifier_type}'. "
        "Choose one of: ['linear', 'mlp']"
    )


def configure_trainable_layers(model: ResNet, strategy: str) -> None:
    """Apply one of the supported ResNet fine-tuning strategies in place."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    if strategy == "classifier":
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        return

    if strategy == "last_block":
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        return

    if strategy == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True
        return

    raise ValueError(
        f"Unknown fine-tuning strategy '{strategy}'. "
        "Choose one of: ['classifier', 'last_block', 'full']"
    )


def set_fine_tuning_mode(model: ResNet, strategy: str) -> None:
    """
    Put trainable modules in training mode and frozen ResNet stages in eval.

    ``requires_grad=False`` prevents gradient updates, but it does not stop
    frozen BatchNorm layers from updating their running statistics when the
    whole model is put in training mode. This function must therefore be called
    at the beginning of every training epoch.
    """
    model.train()

    if strategy == "full":
        return

    frozen_modules: list[nn.Module] = [
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
    ]

    if strategy == "classifier":
        frozen_modules.append(model.layer4)
    elif strategy != "last_block":
        raise ValueError(
            f"Unknown fine-tuning strategy '{strategy}'. "
            "Choose one of: ['classifier', 'last_block', 'full']"
        )

    for module in frozen_modules:
        module.eval()

    if strategy == "last_block":
        model.layer4.train()

    model.fc.train()


def _top_level_trainable_modules(model: ResNet) -> tuple[str, ...]:
    """Return the top-level module names that contain trainable parameters."""
    return tuple(
        sorted(
            {
                parameter_name.split(".", maxsplit=1)[0]
                for parameter_name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
        )
    )


def _build_model_info(
    model: ResNet,
    config: DictConfig,
    classifier_input_features: int,
    pretrained_weights: Any,
) -> ModelInfo:
    """Compute a serializable parameter and architecture summary."""
    total_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    trainable_percentage = (
        100.0 * trainable_parameters / total_parameters
        if total_parameters
        else 0.0
    )

    return ModelInfo(
        model_name=config.model.name,
        pretrained_weights=str(pretrained_weights),
        classifier_type=config.model.classifier_type,
        fine_tuning_strategy=config.model.fine_tuning_strategy,
        classifier_input_features=int(classifier_input_features),
        classifier_output_classes=int(config.model.num_classes),
        total_parameters=int(total_parameters),
        trainable_parameters=int(trainable_parameters),
        trainable_percentage=float(trainable_percentage),
        trainable_modules=_top_level_trainable_modules(model),
    )


def create_model(config: DictConfig, device: torch.device) -> ModelBundle:
    """Create, configure and move the selected pretrained ResNet to device."""
    backbone_spec = get_backbone_spec(config.model.name)
    model = backbone_spec.constructor(weights=backbone_spec.weights)

    classifier_input_features = model.fc.in_features
    model.fc = create_classifier(
        input_features=classifier_input_features,
        num_classes=config.model.num_classes,
        classifier_type=config.model.classifier_type,
        hidden_features=config.model.mlp_hidden_features,
        dropout=config.model.mlp_dropout,
    )

    configure_trainable_layers(
        model=model,
        strategy=config.model.fine_tuning_strategy,
    )
    model.to(device)

    model_info = _build_model_info(
        model=model,
        config=config,
        classifier_input_features=classifier_input_features,
        pretrained_weights=backbone_spec.weights,
    )
    return ModelBundle(model=model, info=model_info)


def print_model_summary(model_bundle: ModelBundle) -> None:
    """Print the most relevant model and fine-tuning information."""
    info = model_bundle.info

    print("\n=== Exercise 2 model preparation ===")
    print(f"Model: {info.model_name}")
    print(f"Pretrained weights: {info.pretrained_weights}")
    print(f"Classifier type: {info.classifier_type}")
    print(f"Fine-tuning strategy: {info.fine_tuning_strategy}")
    print(f"Classifier input features: {info.classifier_input_features}")
    print(f"Classifier output classes: {info.classifier_output_classes}")
    print(f"Trainable modules: {list(info.trainable_modules)}")
    print(f"Total parameters: {info.total_parameters:,}")
    print(f"Trainable parameters: {info.trainable_parameters:,}")
    print(f"Trainable percentage: {info.trainable_percentage:.2f}%")
