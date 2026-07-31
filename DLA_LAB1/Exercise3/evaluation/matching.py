"""Fixed-threshold one-to-one matching for detector diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torchvision.ops import box_iou


@dataclass(frozen=True)
class ClassMatchStats:
    true_positives: int
    false_positives: int
    false_negatives: int
    matched_ious: tuple[float, ...]

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2.0 * self.precision * self.recall / denominator if denominator else 0.0

    @property
    def mean_matched_iou(self) -> float | None:
        if not self.matched_ious:
            return None
        return sum(self.matched_ious) / len(self.matched_ious)

    def to_dict(self) -> dict[str, Any]:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "mean_matched_iou": self.mean_matched_iou,
            "matched_ious": list(self.matched_ious),
        }


@dataclass(frozen=True)
class ImageMatchResult:
    image_id: int
    ground_truth_objects: int
    predictions_above_threshold: int
    true_positives: int
    false_positives: int
    false_negatives: int
    matched_ious: tuple[float, ...]
    empty_target: bool
    per_class: dict[int, ClassMatchStats]

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2.0 * self.precision * self.recall / denominator if denominator else 0.0

    @property
    def mean_matched_iou(self) -> float | None:
        if not self.matched_ious:
            return None
        return sum(self.matched_ious) / len(self.matched_ious)

    def to_flat_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "ground_truth_objects": self.ground_truth_objects,
            "predictions_above_threshold": self.predictions_above_threshold,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "mean_matched_iou": self.mean_matched_iou,
            "empty_target": self.empty_target,
            "false_positive_on_empty_target": bool(
                self.empty_target and self.false_positives > 0
            ),
        }


def _validate_detection_dict(
    value: dict[str, torch.Tensor],
    *,
    prediction: bool,
) -> None:
    required = {"boxes", "labels"}
    if prediction:
        required.add("scores")
    missing = required.difference(value)
    if missing:
        raise KeyError(f"Detection dictionary missing keys: {sorted(missing)}")
    boxes = value["boxes"]
    labels = value["labels"]
    if boxes.ndim != 2 or boxes.shape[1:] != (4,):
        raise ValueError(f"boxes must have shape [N,4], found {tuple(boxes.shape)}")
    if labels.ndim != 1 or labels.shape[0] != boxes.shape[0]:
        raise ValueError("labels must have shape [N] and match boxes.")
    if prediction:
        scores = value["scores"]
        if scores.ndim != 1 or scores.shape[0] != boxes.shape[0]:
            raise ValueError("scores must have shape [N] and match boxes.")


def _match_one_class(
    prediction_boxes: torch.Tensor,
    prediction_scores: torch.Tensor,
    target_boxes: torch.Tensor,
    *,
    iou_threshold: float,
) -> ClassMatchStats:
    prediction_count = int(prediction_boxes.shape[0])
    target_count = int(target_boxes.shape[0])
    if prediction_count == 0:
        return ClassMatchStats(0, 0, target_count, ())
    if target_count == 0:
        return ClassMatchStats(0, prediction_count, 0, ())

    order = torch.argsort(prediction_scores, descending=True)
    prediction_boxes = prediction_boxes[order]
    ious = box_iou(
        prediction_boxes.to(torch.float32),
        target_boxes.to(torch.float32),
    )
    target_used = torch.zeros(target_count, dtype=torch.bool)
    matched_ious: list[float] = []
    true_positives = 0

    for prediction_index in range(prediction_count):
        available = ~target_used
        if not bool(available.any()):
            break
        candidate_ious = ious[prediction_index].clone()
        candidate_ious[~available] = -1.0
        best_iou, best_target = candidate_ious.max(dim=0)
        if float(best_iou.item()) >= iou_threshold:
            true_positives += 1
            target_used[int(best_target.item())] = True
            matched_ious.append(float(best_iou.item()))

    return ClassMatchStats(
        true_positives=true_positives,
        false_positives=prediction_count - true_positives,
        false_negatives=target_count - true_positives,
        matched_ious=tuple(matched_ious),
    )


def match_image_detections(
    prediction: dict[str, torch.Tensor],
    target: dict[str, Any],
    *,
    score_threshold: float,
    iou_threshold: float,
) -> ImageMatchResult:
    """Match predictions to GT one-to-one, independently for every class."""
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be within [0,1].")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be within [0,1].")
    _validate_detection_dict(prediction, prediction=True)
    _validate_detection_dict(target, prediction=False)

    keep = prediction["scores"] >= score_threshold
    prediction_boxes = prediction["boxes"][keep].detach().cpu()
    prediction_labels = prediction["labels"][keep].detach().cpu().to(torch.int64)
    prediction_scores = prediction["scores"][keep].detach().cpu().to(torch.float32)
    target_boxes = target["boxes"].detach().cpu()
    target_labels = target["labels"].detach().cpu().to(torch.int64)

    labels = sorted(
        set(int(value) for value in prediction_labels.tolist())
        | set(int(value) for value in target_labels.tolist())
    )
    per_class: dict[int, ClassMatchStats] = {}
    for label in labels:
        prediction_mask = prediction_labels == label
        target_mask = target_labels == label
        per_class[label] = _match_one_class(
            prediction_boxes[prediction_mask],
            prediction_scores[prediction_mask],
            target_boxes[target_mask],
            iou_threshold=iou_threshold,
        )

    true_positives = sum(item.true_positives for item in per_class.values())
    false_positives = sum(item.false_positives for item in per_class.values())
    false_negatives = sum(item.false_negatives for item in per_class.values())
    matched_ious = tuple(
        value
        for item in per_class.values()
        for value in item.matched_ious
    )
    raw_image_id = target.get("image_id", -1)
    image_id = int(raw_image_id.item()) if torch.is_tensor(raw_image_id) else int(raw_image_id)
    return ImageMatchResult(
        image_id=image_id,
        ground_truth_objects=int(target_boxes.shape[0]),
        predictions_above_threshold=int(prediction_boxes.shape[0]),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        matched_ious=matched_ious,
        empty_target=target_boxes.shape[0] == 0,
        per_class=per_class,
    )


def _metrics_from_counts(
    *,
    true_positives: int,
    false_positives: int,
    false_negatives: int,
    matched_ious: list[float],
) -> dict[str, Any]:
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = true_positives / precision_denominator if precision_denominator else 0.0
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": (
            sum(matched_ious) / len(matched_ious) if matched_ious else None
        ),
    }


def aggregate_match_results(
    results: list[ImageMatchResult],
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Aggregate image-level matching into global and per-class metrics."""
    global_tp = sum(item.true_positives for item in results)
    global_fp = sum(item.false_positives for item in results)
    global_fn = sum(item.false_negatives for item in results)
    global_ious = [value for item in results for value in item.matched_ious]
    global_metrics = _metrics_from_counts(
        true_positives=global_tp,
        false_positives=global_fp,
        false_negatives=global_fn,
        matched_ious=global_ious,
    )
    empty_results = [item for item in results if item.empty_target]
    empty_with_fp = sum(item.false_positives > 0 for item in empty_results)
    global_metrics.update(
        {
            "images": len(results),
            "empty_target_images": len(empty_results),
            "empty_target_images_with_false_positives": empty_with_fp,
            "empty_target_false_positive_rate": (
                empty_with_fp / len(empty_results) if empty_results else None
            ),
        }
    )

    labels = sorted(
        {label for result in results for label in result.per_class}
    )
    per_class: dict[int, dict[str, Any]] = {}
    for label in labels:
        stats = [
            result.per_class[label]
            for result in results
            if label in result.per_class
        ]
        per_class[label] = _metrics_from_counts(
            true_positives=sum(item.true_positives for item in stats),
            false_positives=sum(item.false_positives for item in stats),
            false_negatives=sum(item.false_negatives for item in stats),
            matched_ious=[value for item in stats for value in item.matched_ious],
        )
    return global_metrics, per_class
