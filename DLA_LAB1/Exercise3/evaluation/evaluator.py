"""End-to-end evaluation of a trained Faster R-CNN checkpoint."""

from __future__ import annotations

import csv
import gc
import math
import platform
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, ContextManager

import torch
import torchvision

from Exercise3.data_pipeline.loaders import build_detection_dataloaders
from Exercise3.data_pipeline.loading import (
    DATASET_CONFIGURATION,
    DATASET_REPOSITORY,
    DATASET_REVISION,
    DEFAULT_CACHE_DIR,
    load_detection_dataset,
)
from Exercise3.data_pipeline.taxonomy import (
    NUM_DETECTOR_CLASSES,
    build_detector_label_to_name,
)
from Exercise3.evaluation.configuration import (
    DetectorEvaluationConfig,
    resolve_checkpoint_path,
    resolve_output_directory,
)
from Exercise3.evaluation.matching import (
    aggregate_match_results,
    match_image_detections,
)
from Exercise3.evaluation.metrics import (
    build_map_metric,
    prepare_metric_prediction,
    prepare_metric_target,
    resolve_coco_backend,
    serialize_map_result,
)
from Exercise3.evaluation.reporting import (
    build_markdown_summary,
    save_training_plots,
    write_csv,
    write_json,
)
from Exercise3.evaluation.visualization import (
    save_comparison,
    select_visualization_indices,
)
from Exercise3.models.faster_rcnn import (
    FasterRCNNBaselineConfig,
    build_faster_rcnn_baseline,
)
from Exercise3.training.checkpointing import load_checkpoint
from Exercise3.training.engine import resolve_device, set_reproducibility


def autocast_context(device: torch.device, enabled: bool) -> ContextManager[Any]:
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def _move_images(
    images: list[torch.Tensor],
    device: torch.device,
) -> list[torch.Tensor]:
    return [
        image.to(device, non_blocking=device.type == "cuda")
        for image in images
    ]


def _cpu_prediction(prediction: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in prediction.items()
        if key in {"boxes", "labels", "scores"}
    }


def _checkpoint_model_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    try:
        model_config = checkpoint["config"]["model"]
    except (KeyError, TypeError) as error:
        raise KeyError("Checkpoint does not contain config.model.") from error
    required = {"architecture", "num_classes"}
    missing = required.difference(model_config)
    if missing:
        raise KeyError(f"Checkpoint model config missing: {sorted(missing)}")
    return model_config


def build_model_from_checkpoint(
    checkpoint: dict[str, Any],
    *,
    strict: bool,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    stored = _checkpoint_model_config(checkpoint)
    if int(stored["num_classes"]) != NUM_DETECTOR_CLASSES:
        raise ValueError("Checkpoint taxonomy is incompatible with 44 classes.")

    # Source and trainability do not change the detector architecture.  The
    # complete trained state dict is loaded immediately afterwards, so no
    # external COCO/GTSRB checkpoint is needed during evaluation.
    config = FasterRCNNBaselineConfig(
        architecture=str(stored["architecture"]),
        weights="none",
        num_classes=int(stored["num_classes"]),
        backbone_source="coco",
        gtsrb_checkpoint=None,
        required_gtsrb_strategy=None,
        trainable_backbone="frozen",
        freeze_backbone=True,
        seed=int(checkpoint["config"]["experiment"]["seed"]),
        progress=False,
    )
    model, metadata = build_faster_rcnn_baseline(config)
    incompatibility = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=strict,
    )
    model.to(device)
    model.eval()
    return model, {
        "stored_model_config": stored,
        "missing_keys": list(incompatibility.missing_keys),
        "unexpected_keys": list(incompatibility.unexpected_keys),
        "constructed_without_external_weight_download": True,
        "construction_metadata": metadata,
    }


def _log_evaluation_to_wandb(
    *,
    config: DetectorEvaluationConfig,
    checkpoint: dict[str, Any],
    report: dict[str, Any],
    class_rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    if not config.tracking.enabled:
        return {"enabled": False}
    try:
        import wandb
    except ImportError as error:
        raise ImportError(
            "Evaluation tracking is enabled but wandb is not installed."
        ) from error

    stored_tracking = checkpoint.get("config", {}).get("tracking", {})
    run_id = checkpoint.get("wandb_run_id")
    init_arguments: dict[str, Any] = {
        "project": stored_tracking.get("project", "dla-lab1"),
        "entity": stored_tracking.get("entity"),
        "group": stored_tracking.get("group"),
        "mode": stored_tracking.get("mode", "online"),
        "name": Path(report["checkpoint"]["path"]).parent.name,
    }
    resumed = False
    if config.tracking.resume_training_run:
        if not run_id:
            raise ValueError(
                "Cannot resume W&B evaluation: checkpoint has no wandb_run_id."
            )
        init_arguments.update({"id": run_id, "resume": "allow"})
        resumed = True
    else:
        init_arguments.update(
            {
                "job_type": "faster-rcnn-evaluation",
                "config": config.to_dict(),
                "tags": ["exercise-3-3", "evaluation", report["evaluation"]["split"]],
            }
        )

    run = wandb.init(**init_arguments)
    if run is None:
        raise RuntimeError("wandb.init() did not return a Run object.")

    split = report["evaluation"]["split"]
    standard = report["metrics"]["coco_style"]
    fixed = report["metrics"]["fixed_threshold"]
    summary_values = {
        f"{split}/map_50_95": standard.get("map"),
        f"{split}/map_50": standard.get("map_50"),
        f"{split}/map_75": standard.get("map_75"),
        f"{split}/map_small": standard.get("map_small"),
        f"{split}/map_medium": standard.get("map_medium"),
        f"{split}/map_large": standard.get("map_large"),
        f"{split}/mar_100": standard.get("mar_100"),
        f"{split}/precision_at_0_5": fixed.get("precision"),
        f"{split}/recall_at_0_5": fixed.get("recall"),
        f"{split}/f1_at_0_5": fixed.get("f1"),
        f"{split}/true_positives": fixed.get("true_positives"),
        f"{split}/false_positives": fixed.get("false_positives"),
        f"{split}/false_negatives": fixed.get("false_negatives"),
        f"{split}/inference_seconds": report["evaluation"].get("inference_seconds"),
        f"{split}/images_per_second": report["evaluation"].get("images_per_second"),
    }
    for key, value in summary_values.items():
        if value is not None:
            run.summary[key] = value

    log_payload: dict[str, Any] = {
        f"{split}/evaluation_complete": 1,
        **{key: value for key, value in summary_values.items() if value is not None},
    }
    if config.tracking.log_per_class_table and class_rows:
        columns = list(class_rows[0].keys())
        log_payload[f"{split}/per_class_metrics"] = wandb.Table(
            columns=columns,
            data=[[row.get(column) for column in columns] for row in class_rows],
        )
    if config.tracking.log_images:
        qualitative = report["artifacts"].get("qualitative_visualizations", [])
        if qualitative:
            log_payload[f"{split}/qualitative"] = [
                wandb.Image(item["path"], caption=item.get("reason"))
                for item in qualitative
            ]
        plot_paths = report["artifacts"].get("training_plots", [])
        if plot_paths:
            log_payload["training/plots"] = [wandb.Image(path) for path in plot_paths]
    run.log(log_payload)

    if config.tracking.log_evaluation_artifact:
        artifact = wandb.Artifact(
            name=f"{Path(report['checkpoint']['path']).parent.name}-{split}-evaluation",
            type="evaluation",
            metadata={"split": split},
        )
        for filename in (
            "evaluation_metrics.json",
            "evaluation_summary.md",
            "per_class_metrics.csv",
            "per_image_metrics.csv",
        ):
            path = output_dir / filename
            if path.is_file():
                artifact.add_file(str(path), name=filename)
        run.log_artifact(artifact)

    actual_run_id = str(run.id)
    run.finish(exit_code=0)
    return {
        "enabled": True,
        "run_id": actual_run_id,
        "resumed_training_run": resumed,
        "project": init_arguments["project"],
        "group": init_arguments.get("group"),
    }

def _prediction_counts(
    predictions: list[dict[str, torch.Tensor]],
    threshold: float,
) -> dict[int, int]:
    counts = {label: 0 for label in range(1, NUM_DETECTOR_CLASSES)}
    for prediction in predictions:
        keep = prediction["scores"] >= threshold
        for label in prediction["labels"][keep].tolist():
            counts[int(label)] += 1
    return counts


def _target_counts(targets: list[dict[str, Any]]) -> dict[int, int]:
    counts = {label: 0 for label in range(1, NUM_DETECTOR_CLASSES)}
    for target in targets:
        for label in target["labels"].tolist():
            counts[int(label)] += 1
    return counts


def _build_class_rows(
    *,
    standard_per_class: dict[int, dict[str, float | None]],
    fixed_per_class: dict[int, dict[str, Any]],
    target_counts: dict[int, int],
    prediction_counts: dict[int, int],
) -> list[dict[str, Any]]:
    names = build_detector_label_to_name()
    rows = []
    for label in range(1, NUM_DETECTOR_CLASSES):
        standard = standard_per_class.get(label, {})
        fixed = fixed_per_class.get(label, {})
        rows.append(
            {
                "detector_label": label,
                "gtsrb_class_id": label - 1,
                "class_name": names[label],
                "ground_truth_objects": target_counts[label],
                "predictions_above_fixed_threshold": prediction_counts[label],
                "map_50_95": standard.get("map_50_95"),
                "mar_100": standard.get("mar_100"),
                "true_positives": fixed.get("true_positives", 0),
                "false_positives": fixed.get("false_positives", 0),
                "false_negatives": fixed.get("false_negatives", 0),
                "precision_at_fixed_threshold": fixed.get("precision"),
                "recall_at_fixed_threshold": fixed.get("recall"),
                "f1_at_fixed_threshold": fixed.get("f1"),
                "mean_matched_iou": fixed.get("mean_matched_iou"),
            }
        )
    return rows


def _read_training_summary(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "run_summary.json"
    if not path.is_file():
        return None
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_checkpoint(config: DetectorEvaluationConfig) -> dict[str, Any]:
    checkpoint_path = resolve_checkpoint_path(config)
    output_dir = resolve_output_directory(config, checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.runtime.device)
    amp_enabled = config.runtime.amp and device.type == "cuda"
    set_reproducibility(config.runtime.seed, deterministic=False)

    if device.type == "cuda":
        index = 0 if device.index is None else device.index
        torch.cuda.set_device(index)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(index)

    checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
    model, model_loading = build_model_from_checkpoint(
        checkpoint,
        strict=config.checkpoint.strict,
        device=device,
    )
    original_score_threshold = float(model.roi_heads.score_thresh)
    model.roi_heads.score_thresh = config.evaluation.model_score_threshold

    dataset_dict, resolved_cache_dir = load_detection_dataset(DEFAULT_CACHE_DIR)
    loaders = build_detection_dataloaders(
        dataset_dict,
        train_batch_size=1,
        evaluation_batch_size=config.runtime.batch_size,
        num_workers=config.runtime.num_workers,
        pin_memory=config.runtime.pin_memory and device.type == "cuda",
        persistent_workers=config.runtime.persistent_workers,
        seed=config.runtime.seed,
    )
    loader = (
        loaders.validation
        if config.evaluation.split == "validation"
        else loaders.test
    )
    dataset = (
        loaders.datasets.validation
        if config.evaluation.split == "validation"
        else loaders.datasets.test
    )

    backend = resolve_coco_backend(config.evaluation.backend)
    map_metric = build_map_metric(
        backend=backend,
        class_metrics=config.evaluation.class_metrics,
    )

    predictions_cpu: list[dict[str, torch.Tensor]] = []
    targets_cpu: list[dict[str, Any]] = []
    per_image_rows: list[dict[str, Any]] = []
    match_results = []
    images_evaluated = 0
    objects_evaluated = 0
    inference_seconds = 0.0
    dataset_index = 0

    model.eval()
    with torch.inference_mode():
        for batch_index, (images, targets) in enumerate(loader):
            if (
                config.evaluation.max_batches is not None
                and batch_index >= config.evaluation.max_batches
            ):
                break
            moved_images = _move_images(images, device)
            start = time.perf_counter()
            with autocast_context(device, amp_enabled):
                batch_predictions = model(moved_images)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - start

            batch_predictions_cpu = [
                _cpu_prediction(item) for item in batch_predictions
            ]
            batch_targets_cpu = [
                {
                    key: value.detach().cpu() if torch.is_tensor(value) else value
                    for key, value in target.items()
                }
                for target in targets
            ]
            map_metric.update(
                [prepare_metric_prediction(item) for item in batch_predictions_cpu],
                [prepare_metric_target(item) for item in batch_targets_cpu],
            )

            for prediction, target in zip(
                batch_predictions_cpu,
                batch_targets_cpu,
                strict=True,
            ):
                match = match_image_detections(
                    prediction,
                    target,
                    score_threshold=config.evaluation.fixed_score_threshold,
                    iou_threshold=config.evaluation.fixed_iou_threshold,
                )
                row = {
                    "dataset_index": dataset_index,
                    **match.to_flat_dict(),
                    "maximum_prediction_score": (
                        float(prediction["scores"].max().item())
                        if prediction["scores"].numel()
                        else None
                    ),
                }
                per_image_rows.append(row)
                match_results.append(match)
                predictions_cpu.append(prediction)
                targets_cpu.append(target)
                images_evaluated += 1
                objects_evaluated += int(target["boxes"].shape[0])
                dataset_index += 1

    standard = serialize_map_result(map_metric.compute())
    fixed_global, fixed_per_class = aggregate_match_results(match_results)
    target_counts = _target_counts(targets_cpu)
    prediction_counts = _prediction_counts(
        predictions_cpu,
        config.evaluation.fixed_score_threshold,
    )
    class_rows = _build_class_rows(
        standard_per_class=standard.pop("per_class"),
        fixed_per_class=fixed_per_class,
        target_counts=target_counts,
        prediction_counts=prediction_counts,
    )

    write_csv(
        output_dir / "per_image_metrics.csv",
        per_image_rows,
        fieldnames=list(per_image_rows[0].keys()) if per_image_rows else [],
    )
    write_csv(
        output_dir / "per_class_metrics.csv",
        class_rows,
        fieldnames=list(class_rows[0].keys()),
    )
    if config.evaluation.save_predictions:
        torch.save(
            {
                "format_version": 1,
                "split": config.evaluation.split,
                "checkpoint": str(checkpoint_path),
                "predictions": predictions_cpu,
                "targets": targets_cpu,
            },
            output_dir / "predictions.pt",
        )

    visualization_records: list[dict[str, Any]] = []
    if config.visualization.enabled:
        selected = select_visualization_indices(
            per_image_rows,
            sample_count=config.visualization.samples,
            seed=config.runtime.seed,
        )
        qualitative_dir = output_dir / "qualitative"
        for index, reason in selected:
            image, target = dataset[index]
            prediction = predictions_cpu[index]
            image_id = int(target["image_id"])
            path = save_comparison(
                image=image,
                target=target,
                prediction=prediction,
                output_path=(
                    qualitative_dir
                    / f"{config.evaluation.split}_{index:04d}_image-{image_id}_{reason}.png"
                ),
                score_threshold=config.visualization.score_threshold,
                box_width=config.visualization.box_width,
                font_path=config.visualization.font_path,
                font_size=config.visualization.font_size,
                title=f"{config.evaluation.split} index {index}, image {image_id}",
            )
            visualization_records.append(
                {
                    "dataset_index": index,
                    "image_id": image_id,
                    "reason": reason,
                    "path": str(path),
                }
            )

    plots = (
        save_training_plots(checkpoint_path.parent, output_dir / "training_plots")
        if config.output.save_training_plots
        else []
    )
    cuda_peak = None
    if device.type == "cuda":
        index = 0 if device.index is None else device.index
        cuda_peak = {
            "device_name": torch.cuda.get_device_name(index),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
        }

    report = {
        "runtime": {
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "device": str(device),
            "amp_enabled": amp_enabled,
            "cuda_peak_memory": cuda_peak,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "epoch": int(checkpoint["epoch"]),
            "global_step": int(checkpoint["global_step"]),
            "best_epoch": int(checkpoint["best_epoch"]),
            "best_validation_total_loss": float(checkpoint["best_metric"]),
            "model_loading": model_loading,
        },
        "dataset": {
            "repository": DATASET_REPOSITORY,
            "configuration": DATASET_CONFIGURATION,
            "revision": DATASET_REVISION,
            "resolved_cache_dir": str(resolved_cache_dir),
            "split": config.evaluation.split,
            "split_images": len(dataset),
            "images_evaluated": images_evaluated,
            "objects_evaluated": objects_evaluated,
        },
        "evaluation": {
            "split": config.evaluation.split,
            "backend": backend,
            "model_score_threshold": config.evaluation.model_score_threshold,
            "fixed_score_threshold": config.evaluation.fixed_score_threshold,
            "fixed_iou_threshold": config.evaluation.fixed_iou_threshold,
            "max_batches": config.evaluation.max_batches,
            "scientific_full_evaluation": config.evaluation.max_batches is None,
            "inference_seconds": inference_seconds,
            "images_per_second": (
                images_evaluated / inference_seconds if inference_seconds else None
            ),
            "original_model_score_threshold": original_score_threshold,
        },
        "metrics": {
            "coco_style": standard,
            "fixed_threshold": fixed_global,
        },
        "training_summary": _read_training_summary(checkpoint_path.parent),
        "artifacts": {
            "output_directory": str(output_dir),
            "per_image_metrics": str(output_dir / "per_image_metrics.csv"),
            "per_class_metrics": str(output_dir / "per_class_metrics.csv"),
            "predictions": (
                str(output_dir / "predictions.pt")
                if config.evaluation.save_predictions
                else None
            ),
            "training_plots": plots,
            "qualitative_visualizations": visualization_records,
        },
    }
    write_json(output_dir / "evaluation_metrics.json", report)
    if config.output.save_markdown_summary:
        (output_dir / "evaluation_summary.md").write_text(
            build_markdown_summary(report),
            encoding="utf-8",
        )

    report["tracking"] = _log_evaluation_to_wandb(
        config=config,
        checkpoint=checkpoint,
        report=report,
        class_rows=class_rows,
        output_dir=output_dir,
    )
    write_json(output_dir / "evaluation_metrics.json", report)

    model.roi_heads.score_thresh = original_score_threshold
    del model, map_metric
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report
