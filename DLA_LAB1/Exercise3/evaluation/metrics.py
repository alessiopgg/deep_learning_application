"""TorchMetrics/COCO metric helpers for Exercise 3 detector evaluation."""

from __future__ import annotations

import importlib.util
from typing import Any

import torch


def resolve_coco_backend(requested: str) -> str:
    available = {
        "pycocotools": importlib.util.find_spec("pycocotools") is not None,
        "faster_coco_eval": importlib.util.find_spec("faster_coco_eval") is not None,
    }
    if requested == "auto":
        if available["pycocotools"]:
            return "pycocotools"
        if available["faster_coco_eval"]:
            return "faster_coco_eval"
        raise ImportError(
            "Install pycocotools or faster-coco-eval for COCO-style metrics."
        )
    if requested not in available:
        raise ValueError(
            "backend must be auto, pycocotools or faster_coco_eval."
        )
    if not available[requested]:
        package = "faster-coco-eval" if requested == "faster_coco_eval" else requested
        raise ImportError(f"Requested COCO backend is not installed: {package}")
    return requested


def build_map_metric(*, backend: str, class_metrics: bool):
    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
    except ImportError as error:
        raise ImportError("torchmetrics is required for detector evaluation.") from error
    return MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=class_metrics,
        backend=backend,
    )


def prepare_metric_prediction(
    prediction: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        "boxes": prediction["boxes"].detach().cpu().to(torch.float32),
        "scores": prediction["scores"].detach().cpu().to(torch.float32),
        "labels": prediction["labels"].detach().cpu().to(torch.int64),
    }


def prepare_metric_target(target: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        "boxes": target["boxes"].detach().cpu().to(torch.float32),
        "labels": target["labels"].detach().cpu().to(torch.int64),
    }


def _scalar(value: Any) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("Expected a scalar metric tensor.")
        result = float(value.detach().cpu().item())
    else:
        result = float(value)
    # TorchMetrics uses -1 when a COCO metric is undefined.
    return None if result < 0 else result


def serialize_map_result(result: dict[str, Any]) -> dict[str, Any]:
    scalar_keys = (
        "map",
        "map_50",
        "map_75",
        "map_small",
        "map_medium",
        "map_large",
        "mar_1",
        "mar_10",
        "mar_100",
        "mar_small",
        "mar_medium",
        "mar_large",
    )
    serialized: dict[str, Any] = {
        key: _scalar(result.get(key)) for key in scalar_keys
    }

    classes = result.get("classes")
    map_per_class = result.get("map_per_class")
    mar_per_class = result.get("mar_100_per_class")
    per_class: dict[int, dict[str, float | None]] = {}
    if torch.is_tensor(classes) and classes.numel() > 0:
        class_values = classes.detach().cpu().to(torch.int64).tolist()
        map_values = (
            map_per_class.detach().cpu().tolist()
            if torch.is_tensor(map_per_class)
            else []
        )
        mar_values = (
            mar_per_class.detach().cpu().tolist()
            if torch.is_tensor(mar_per_class)
            else []
        )
        for index, label in enumerate(class_values):
            per_class[int(label)] = {
                "map_50_95": (
                    _scalar(map_values[index]) if index < len(map_values) else None
                ),
                "mar_100": (
                    _scalar(mar_values[index]) if index < len(mar_values) else None
                ),
            }
    serialized["per_class"] = per_class
    return serialized
