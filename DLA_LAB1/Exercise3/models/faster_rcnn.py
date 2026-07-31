"""Faster R-CNN baseline construction for Exercise 3.3.

Step 9 builds the model only. It does not run a forward pass, compute losses,
or update parameters. The baseline starts from Torchvision's COCO-pretrained
Faster R-CNN ResNet-50-FPN, replaces the COCO box predictor with a new
44-class predictor, and freezes the complete backbone including the FPN.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
from torch import nn
from torchvision.models.detection import (
    FasterRCNN,
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from Exercise3.data_pipeline.taxonomy import NUM_DETECTOR_CLASSES


WeightsChoice = Literal["coco", "none"]


@dataclass(frozen=True)
class FasterRCNNBaselineConfig:
    """Configuration of the initial Faster R-CNN baseline."""

    architecture: str = "fasterrcnn_resnet50_fpn"
    weights: WeightsChoice = "coco"
    num_classes: int = NUM_DETECTOR_CLASSES
    freeze_backbone: bool = True
    seed: int = 42
    progress: bool = True

    def validate(self) -> None:
        if self.architecture != "fasterrcnn_resnet50_fpn":
            raise ValueError(
                "Step 9 supports only 'fasterrcnn_resnet50_fpn'."
            )
        if self.weights not in {"coco", "none"}:
            raise ValueError("weights must be either 'coco' or 'none'.")
        if self.num_classes != NUM_DETECTOR_CLASSES:
            raise ValueError(
                "The current verified taxonomy requires exactly "
                f"{NUM_DETECTOR_CLASSES} detector classes including background."
            )
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParameterCount:
    """Total, trainable and frozen parameter counts for one component."""

    total: int
    trainable: int
    frozen: int

    @property
    def trainable_percentage(self) -> float:
        if self.total == 0:
            return 0.0
        return 100.0 * self.trainable / self.total

    def to_dict(self) -> dict[str, int | float]:
        return {
            "total": self.total,
            "trainable": self.trainable,
            "frozen": self.frozen,
            "trainable_percentage": self.trainable_percentage,
        }


def _count_parameters(module: nn.Module) -> ParameterCount:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    return ParameterCount(
        total=total,
        trainable=trainable,
        frozen=total - trainable,
    )


def _tensor_sha256(tensor: torch.Tensor) -> str:
    """Return a stable checksum for a representative model tensor."""
    contiguous = tensor.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def _resolve_weights(
    choice: WeightsChoice,
) -> FasterRCNN_ResNet50_FPN_Weights | None:
    if choice == "coco":
        return FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    if choice == "none":
        return None
    raise ValueError(f"Unsupported weights choice: {choice}")


def freeze_complete_backbone(model: FasterRCNN) -> None:
    """Freeze both the ResNet body and all FPN parameters."""
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False


def configure_model_for_training(
    model: FasterRCNN,
    *,
    freeze_backbone: bool,
) -> None:
    """Set train mode while keeping a frozen backbone in evaluation mode.

    Calling ``model.train()`` recursively marks every child module as training.
    This helper must therefore be called at the start of each training epoch if
    the backbone is frozen.
    """
    model.train()
    if freeze_backbone:
        model.backbone.eval()


def build_faster_rcnn_baseline(
    config: FasterRCNNBaselineConfig | None = None,
) -> tuple[FasterRCNN, dict[str, Any]]:
    """Build the Step 9 baseline and return construction metadata."""
    resolved_config = config or FasterRCNNBaselineConfig()
    resolved_config.validate()

    torch.manual_seed(resolved_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved_config.seed)

    weights = _resolve_weights(resolved_config.weights)

    # Do not pass num_classes when loading COCO weights: the official builder
    # first reconstructs the COCO predictor. We replace it explicitly below.
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=None,
        progress=resolved_config.progress,
    )

    old_predictor = model.roi_heads.box_predictor
    old_num_classes = int(old_predictor.cls_score.out_features)
    predictor_input_features = int(old_predictor.cls_score.in_features)

    # Re-seed immediately before the randomly initialized traffic-sign head so
    # its initialization is reproducible independently of constructor details.
    torch.manual_seed(resolved_config.seed)
    model.roi_heads.box_predictor = FastRCNNPredictor(
        predictor_input_features,
        resolved_config.num_classes,
    )

    if resolved_config.freeze_backbone:
        freeze_complete_backbone(model)

    configure_model_for_training(
        model,
        freeze_backbone=resolved_config.freeze_backbone,
    )

    metadata = {
        "architecture": resolved_config.architecture,
        "weights": (
            "FasterRCNN_ResNet50_FPN_Weights.DEFAULT"
            if weights is not None
            else "none"
        ),
        "weights_enum_name": weights.name if weights is not None else None,
        "weights_url": weights.url if weights is not None else None,
        "original_predictor_num_classes": old_num_classes,
        "new_predictor_num_classes": int(
            model.roi_heads.box_predictor.cls_score.out_features
        ),
        "predictor_input_features": predictor_input_features,
        "bbox_regression_outputs": int(
            model.roi_heads.box_predictor.bbox_pred.out_features
        ),
        "background_label": 0,
        "foreground_label_range": [1, resolved_config.num_classes - 1],
        "backbone_frozen": resolved_config.freeze_backbone,
        "backbone_training_mode": bool(model.backbone.training),
        "representative_backbone_tensor": "backbone.body.conv1.weight",
        "representative_backbone_sha256": _tensor_sha256(
            model.backbone.body.conv1.weight
        ),
        "config": resolved_config.to_dict(),
    }
    return model, metadata


def summarize_faster_rcnn(
    model: FasterRCNN,
    construction_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build a serializable audit of parameters and trainability."""
    components: dict[str, nn.Module] = {
        "backbone": model.backbone,
        "rpn": model.rpn,
        "roi_heads.box_head": model.roi_heads.box_head,
        "roi_heads.box_predictor": model.roi_heads.box_predictor,
    }

    component_counts = {
        name: _count_parameters(module)
        for name, module in components.items()
    }
    complete_count = _count_parameters(model)

    trainable_components: list[str] = []
    frozen_components: list[str] = []
    partially_trainable_components: list[str] = []

    for name, count in component_counts.items():
        if count.total == 0:
            continue
        if count.trainable == 0:
            frozen_components.append(name)
        elif count.trainable == count.total:
            trainable_components.append(name)
        else:
            partially_trainable_components.append(name)

    named_trainable_parameters = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    named_frozen_parameters = [
        name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    ]

    report = {
        **construction_metadata,
        "parameters": {
            "model": complete_count.to_dict(),
            "components": {
                name: count.to_dict()
                for name, count in component_counts.items()
            },
            "trainable_components": trainable_components,
            "frozen_components": frozen_components,
            "partially_trainable_components": partially_trainable_components,
            "trainable_parameter_tensor_count": len(named_trainable_parameters),
            "frozen_parameter_tensor_count": len(named_frozen_parameters),
            "first_trainable_parameter_names": named_trainable_parameters[:20],
            "first_frozen_parameter_names": named_frozen_parameters[:20],
        },
        "model_modes": {
            "model_training": bool(model.training),
            "backbone_training": bool(model.backbone.training),
            "rpn_training": bool(model.rpn.training),
            "roi_heads_training": bool(model.roi_heads.training),
        },
    }

    _validate_model_report(report)
    return report


def _validate_model_report(report: dict[str, Any]) -> None:
    """Reject silent mistakes in head replacement or freezing policy."""
    expected_classes = NUM_DETECTOR_CLASSES
    if report["new_predictor_num_classes"] != expected_classes:
        raise ValueError(
            "The Faster R-CNN classification head has the wrong output size."
        )
    if report["bbox_regression_outputs"] != expected_classes * 4:
        raise ValueError(
            "The Faster R-CNN box regressor must output four values per class."
        )

    parameters = report["parameters"]
    components = parameters["components"]
    if report["backbone_frozen"]:
        if components["backbone"]["trainable"] != 0:
            raise ValueError("The complete backbone was not frozen.")
        if report["model_modes"]["backbone_training"]:
            raise ValueError(
                "A frozen backbone must remain in evaluation mode."
            )

    for required_trainable in (
        "rpn",
        "roi_heads.box_head",
        "roi_heads.box_predictor",
    ):
        if components[required_trainable]["trainable"] == 0:
            raise ValueError(
                f"Required component '{required_trainable}' has no trainable parameters."
            )

    if parameters["model"]["trainable"] <= 0:
        raise ValueError("The model has no trainable parameters.")
