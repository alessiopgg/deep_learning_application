"""Faster R-CNN construction and trainability policies for Exercise 3.3."""

from __future__ import annotations

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
from Exercise3.models.gtsrb_transfer import (
    load_gtsrb_backbone,
    resolve_project_path,
    tensor_sha256,
)


WeightsChoice = Literal["coco", "none"]
BackboneSource = Literal["coco", "gtsrb"]
TrainableBackbone = Literal["frozen", "layer4", "layer3_layer4"]


@dataclass(frozen=True)
class FasterRCNNBaselineConfig:
    """Configuration shared by the COCO baseline and the A-D matrix."""

    architecture: str = "fasterrcnn_resnet50_fpn"
    weights: WeightsChoice = "coco"
    num_classes: int = NUM_DETECTOR_CLASSES
    backbone_source: BackboneSource = "coco"
    gtsrb_checkpoint: str | None = None
    required_gtsrb_strategy: str | None = "full"
    trainable_backbone: TrainableBackbone = "frozen"
    # Backward-compatible alias used by the Step 9/10/11 code and YAML.
    # It must agree with trainable_backbone when explicitly supplied.
    freeze_backbone: bool | None = None
    seed: int = 42
    progress: bool = True

    @property
    def backbone_is_frozen(self) -> bool:
        return self.trainable_backbone == "frozen"

    def validate(self) -> None:
        if self.architecture != "fasterrcnn_resnet50_fpn":
            raise ValueError(
                "Only 'fasterrcnn_resnet50_fpn' is supported."
            )
        if self.weights not in {"coco", "none"}:
            raise ValueError("weights must be either 'coco' or 'none'.")
        if self.num_classes != NUM_DETECTOR_CLASSES:
            raise ValueError(
                "The verified taxonomy requires exactly "
                f"{NUM_DETECTOR_CLASSES} detector classes including "
                "background."
            )
        if self.backbone_source not in {"coco", "gtsrb"}:
            raise ValueError("backbone_source must be coco or gtsrb.")
        if self.trainable_backbone not in {
            "frozen",
            "layer4",
            "layer3_layer4",
        }:
            raise ValueError(
                "trainable_backbone must be frozen, layer4 or "
                "layer3_layer4."
            )
        if self.freeze_backbone is not None:
            expected = self.trainable_backbone == "frozen"
            if bool(self.freeze_backbone) != expected:
                raise ValueError(
                    "freeze_backbone conflicts with trainable_backbone. "
                    "Use freeze_backbone=true only with "
                    "trainable_backbone=frozen."
                )
        if self.backbone_source == "gtsrb":
            if not self.gtsrb_checkpoint:
                raise ValueError(
                    "backbone_source=gtsrb requires "
                    "model.gtsrb_checkpoint."
                )
            if self.weights != "coco":
                raise ValueError(
                    "GTSRB runs must start from the COCO detector so FPN, "
                    "RPN and RoI heads keep the same initialization as Run A."
                )
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParameterCount:
    total: int
    trainable: int
    frozen: int

    @property
    def trainable_percentage(self) -> float:
        return (
            0.0
            if self.total == 0
            else 100.0 * self.trainable / self.total
        )

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


def _resolve_weights(
    choice: WeightsChoice,
) -> FasterRCNN_ResNet50_FPN_Weights | None:
    if choice == "coco":
        return FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    if choice == "none":
        return None
    raise ValueError(f"Unsupported weights choice: {choice}")


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = enabled


def freeze_complete_backbone(model: FasterRCNN) -> None:
    _set_requires_grad(model.backbone, False)


def configure_backbone_trainability(
    model: FasterRCNN,
    policy: TrainableBackbone,
) -> None:
    """Apply the frozen/layer4/layer3+layer4 policy to body and FPN."""

    freeze_complete_backbone(model)

    if policy == "frozen":
        return
    if policy == "layer4":
        _set_requires_grad(model.backbone.body.layer4, True)
        _set_requires_grad(model.backbone.fpn, True)
        return
    if policy == "layer3_layer4":
        _set_requires_grad(model.backbone.body.layer3, True)
        _set_requires_grad(model.backbone.body.layer4, True)
        _set_requires_grad(model.backbone.fpn, True)
        return

    raise ValueError(f"Unsupported trainable_backbone policy: {policy}")


def configure_model_for_training(
    model: FasterRCNN,
    *,
    trainable_backbone: TrainableBackbone | None = None,
    freeze_backbone: bool | None = None,
) -> None:
    """Set detector train mode while frozen ResNet stages remain in eval."""

    if trainable_backbone is None:
        if freeze_backbone is None or freeze_backbone:
            policy: TrainableBackbone = "frozen"
        else:
            raise ValueError(
                "freeze_backbone=False is ambiguous. Pass "
                "trainable_backbone=layer4 or layer3_layer4 explicitly."
            )
    else:
        policy = trainable_backbone
        if (
            freeze_backbone is not None
            and freeze_backbone != (policy == "frozen")
        ):
            raise ValueError(
                "freeze_backbone conflicts with trainable_backbone."
            )

    model.train()
    body = model.backbone.body

    if policy == "frozen":
        model.backbone.eval()
        return

    # Start from eval for every body stage, then enable only trainable stages.
    body.eval()
    model.backbone.fpn.train()

    if policy == "layer4":
        body.layer4.train()
        return
    if policy == "layer3_layer4":
        body.layer3.train()
        body.layer4.train()
        return

    raise ValueError(f"Unsupported trainable_backbone policy: {policy}")


def build_faster_rcnn_baseline(
    config: FasterRCNNBaselineConfig | None = None,
) -> tuple[FasterRCNN, dict[str, Any]]:
    """Construct one detector and apply the requested backbone policy."""

    resolved_config = config or FasterRCNNBaselineConfig()
    resolved_config.validate()

    torch.manual_seed(resolved_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved_config.seed)

    weights = _resolve_weights(resolved_config.weights)
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=None,
        progress=resolved_config.progress,
    )

    old_predictor = model.roi_heads.box_predictor
    old_num_classes = int(old_predictor.cls_score.out_features)
    predictor_input_features = int(
        old_predictor.cls_score.in_features
    )

    torch.manual_seed(resolved_config.seed)
    model.roi_heads.box_predictor = FastRCNNPredictor(
        predictor_input_features,
        resolved_config.num_classes,
    )

    gtsrb_transfer: dict[str, Any] | None = None
    if resolved_config.backbone_source == "gtsrb":
        checkpoint_path = resolve_project_path(
            str(resolved_config.gtsrb_checkpoint)
        ).resolve()
        gtsrb_transfer = load_gtsrb_backbone(
            model,
            checkpoint_path=checkpoint_path,
            required_strategy=resolved_config.required_gtsrb_strategy,
        )

    configure_backbone_trainability(
        model,
        resolved_config.trainable_backbone,
    )
    configure_model_for_training(
        model,
        trainable_backbone=resolved_config.trainable_backbone,
    )

    metadata = {
        "architecture": resolved_config.architecture,
        "weights": (
            "FasterRCNN_ResNet50_FPN_Weights.DEFAULT"
            if weights is not None
            else "none"
        ),
        "weights_enum_name": (
            weights.name if weights is not None else None
        ),
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
        "foreground_label_range": [
            1,
            resolved_config.num_classes - 1,
        ],
        "backbone_source": resolved_config.backbone_source,
        "trainable_backbone": resolved_config.trainable_backbone,
        "backbone_frozen": resolved_config.backbone_is_frozen,
        "backbone_training_mode": bool(model.backbone.training),
        "representative_backbone_tensor": (
            "backbone.body.conv1.weight"
        ),
        "representative_backbone_sha256": tensor_sha256(
            model.backbone.body.conv1.weight
        ),
        "gtsrb_transfer": gtsrb_transfer,
        "config": resolved_config.to_dict(),
    }
    return model, metadata


def summarize_faster_rcnn(
    model: FasterRCNN,
    construction_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build and validate a parameter/mode audit for one detector."""

    components: dict[str, nn.Module] = {
        "backbone": model.backbone,
        "backbone.body.layer3": model.backbone.body.layer3,
        "backbone.body.layer4": model.backbone.body.layer4,
        "backbone.fpn": model.backbone.fpn,
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

    trainable_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    frozen_names = [
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
            "partially_trainable_components": (
                partially_trainable_components
            ),
            "trainable_parameter_tensor_count": len(trainable_names),
            "frozen_parameter_tensor_count": len(frozen_names),
            "first_trainable_parameter_names": trainable_names[:20],
            "first_frozen_parameter_names": frozen_names[:20],
        },
        "model_modes": {
            "model_training": bool(model.training),
            "backbone_training": bool(model.backbone.training),
            "backbone_body_training": bool(
                model.backbone.body.training
            ),
            "layer3_training": bool(
                model.backbone.body.layer3.training
            ),
            "layer4_training": bool(
                model.backbone.body.layer4.training
            ),
            "fpn_training": bool(model.backbone.fpn.training),
            "rpn_training": bool(model.rpn.training),
            "roi_heads_training": bool(model.roi_heads.training),
        },
    }
    _validate_model_report(report)
    return report


def _validate_model_report(report: dict[str, Any]) -> None:
    expected_classes = NUM_DETECTOR_CLASSES

    if report["new_predictor_num_classes"] != expected_classes:
        raise ValueError(
            "The classification head has the wrong output size."
        )
    if report["bbox_regression_outputs"] != expected_classes * 4:
        raise ValueError(
            "The box regressor must output four values per class."
        )

    components = report["parameters"]["components"]
    policy = report["trainable_backbone"]

    if policy == "frozen":
        if components["backbone"]["trainable"] != 0:
            raise ValueError("The complete backbone was not frozen.")
        if report["model_modes"]["backbone_training"]:
            raise ValueError(
                "A frozen backbone must remain in evaluation mode."
            )
    elif policy == "layer4":
        if components["backbone.body.layer3"]["trainable"] != 0:
            raise ValueError(
                "layer3 must be frozen for the layer4 policy."
            )
        if components["backbone.body.layer4"]["trainable"] == 0:
            raise ValueError(
                "layer4 must be trainable for the layer4 policy."
            )
        if components["backbone.fpn"]["trainable"] == 0:
            raise ValueError(
                "FPN must be trainable for the layer4 policy."
            )
    elif policy == "layer3_layer4":
        for name in (
            "backbone.body.layer3",
            "backbone.body.layer4",
            "backbone.fpn",
        ):
            if components[name]["trainable"] == 0:
                raise ValueError(f"{name} must be trainable.")

    for required in (
        "rpn",
        "roi_heads.box_head",
        "roi_heads.box_predictor",
    ):
        if components[required]["trainable"] == 0:
            raise ValueError(
                f"Required component {required!r} is not trainable."
            )

    if report["parameters"]["model"]["trainable"] <= 0:
        raise ValueError("The model has no trainable parameters.")
