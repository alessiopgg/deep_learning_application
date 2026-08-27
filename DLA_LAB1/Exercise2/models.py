"""ResNet construction and fine-tuning policies for Exercise 2."""

import torch
from omegaconf import DictConfig
from torch import nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    resnet18,
    resnet50,
)


BACKBONES = {
    "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V2),
}


def create_input_transform(model_name: str):
    return BACKBONES[model_name][1].transforms()


def configure_trainable_layers(model, strategy: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    if strategy == "classifier":
        modules = (model.fc,)
    elif strategy == "last_block":
        modules = (model.layer4, model.fc)
    elif strategy == "full":
        modules = (model,)
    else:
        raise ValueError(f"Unknown fine-tuning strategy: {strategy}")

    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True


def set_fine_tuning_mode(model, strategy: str) -> None:
    """Keep frozen BatchNorm stages in eval mode during selective fine-tuning."""
    model.train()
    if strategy == "full":
        return

    frozen = [
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
    ]
    if strategy == "classifier":
        frozen.append(model.layer4)

    for module in frozen:
        module.eval()
    model.fc.train()


def create_model(config: DictConfig, device: torch.device):
    constructor, weights = BACKBONES[config.model.name]
    model = constructor(weights=weights)
    input_features = model.fc.in_features

    if config.model.classifier_type == "linear":
        model.fc = nn.Linear(input_features, config.model.num_classes)
    else:
        model.fc = nn.Sequential(
            nn.Linear(input_features, config.model.mlp_hidden_features),
            nn.ReLU(),
            nn.Dropout(config.model.mlp_dropout),
            nn.Linear(config.model.mlp_hidden_features, config.model.num_classes),
        )

    configure_trainable_layers(model, config.model.fine_tuning_strategy)
    model.to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info = {
        "model_name": config.model.name,
        "pretrained_weights": str(weights),
        "classifier_type": config.model.classifier_type,
        "fine_tuning_strategy": config.model.fine_tuning_strategy,
        "classifier_input_features": int(input_features),
        "classifier_output_classes": int(config.model.num_classes),
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "trainable_percentage": float(100 * trainable / total),
        "trainable_modules": sorted(
            {
                name.split(".", 1)[0]
                for name, p in model.named_parameters()
                if p.requires_grad
            }
        ),
    }
    return model, info


def print_model_summary(info: dict) -> None:
    print("\n=== Exercise 2 model preparation ===")
    print(f"Model: {info['model_name']}")
    print(f"Pretrained weights: {info['pretrained_weights']}")
    print(f"Classifier type: {info['classifier_type']}")
    print(f"Fine-tuning strategy: {info['fine_tuning_strategy']}")
    print(f"Classifier input features: {info['classifier_input_features']}")
    print(f"Classifier output classes: {info['classifier_output_classes']}")
    print(f"Trainable modules: {info['trainable_modules']}")
    print(f"Total parameters: {info['total_parameters']:,}")
    print(f"Trainable parameters: {info['trainable_parameters']:,}")
    print(f"Trainable percentage: {info['trainable_percentage']:.2f}%")
