"""Aggregate completed detector runs into local and W&B comparisons."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


COMPARISON_FIELDS = (
    "experiment_id",
    "experiment_name",
    "status",
    "run_name",
    "run_dir",
    "wandb_run_id",
    "backbone_source",
    "trainable_backbone",
    "trainable_parameters",
    "detector_learning_rate",
    "backbone_learning_rate",
    "epochs",
    "best_epoch",
    "best_validation_total_loss",
    "training_seconds",
    "peak_training_allocated_mib",
    "map_50_95",
    "map_50",
    "map_75",
    "map_small",
    "map_medium",
    "map_large",
    "mar_100",
    "precision_at_0_5",
    "recall_at_0_5",
    "f1_at_0_5",
    "true_positives",
    "false_positives",
    "false_negatives",
    "inference_seconds",
    "images_per_second",
)


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required result file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def build_comparison_row(
    *,
    experiment_id: str,
    experiment_name: str,
    run_dir: Path,
) -> dict[str, Any]:
    config = _read_json_object(run_dir / "config.json")
    runtime = _read_json_object(run_dir / "runtime_metadata.json")
    summary = _read_json_object(run_dir / "run_summary.json")
    evaluation = _read_json_object(
        run_dir / "evaluation" / "validation" / "evaluation_metrics.json"
    )
    standard = evaluation["metrics"]["coco_style"]
    fixed = evaluation["metrics"]["fixed_threshold"]
    model_runtime = runtime["model"]
    model_parameters = model_runtime["parameters"]["model"]

    history = _read_json(run_dir / "history.json")
    peak_allocated = None
    if isinstance(history, list) and history:
        values = [
            row.get("train_peak_allocated_bytes")
            for row in history
            if row.get("train_peak_allocated_bytes") is not None
        ]
        peak_allocated = max(values) if values else None

    return {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "status": "completed",
        "run_name": summary["run_name"],
        "run_dir": str(run_dir),
        "wandb_run_id": summary.get("wandb_run_id"),
        "backbone_source": config["model"]["backbone_source"],
        "trainable_backbone": config["model"]["trainable_backbone"],
        "trainable_parameters": model_parameters["trainable"],
        "detector_learning_rate": config["optimizer"]["learning_rate"],
        "backbone_learning_rate": (
            config["optimizer"]["backbone_learning_rate"]
            if config["model"]["trainable_backbone"] != "frozen"
            else None
        ),
        "epochs": config["training"]["epochs"],
        "best_epoch": summary["best_epoch"],
        "best_validation_total_loss": summary["best_validation_total_loss"],
        "training_seconds": summary["duration_seconds"],
        "peak_training_allocated_mib": (
            peak_allocated / (1024**2) if peak_allocated is not None else None
        ),
        "map_50_95": standard.get("map"),
        "map_50": standard.get("map_50"),
        "map_75": standard.get("map_75"),
        "map_small": standard.get("map_small"),
        "map_medium": standard.get("map_medium"),
        "map_large": standard.get("map_large"),
        "mar_100": standard.get("mar_100"),
        "precision_at_0_5": fixed.get("precision"),
        "recall_at_0_5": fixed.get("recall"),
        "f1_at_0_5": fixed.get("f1"),
        "true_positives": fixed.get("true_positives"),
        "false_positives": fixed.get("false_positives"),
        "false_negatives": fixed.get("false_negatives"),
        "inference_seconds": evaluation["evaluation"].get("inference_seconds"),
        "images_per_second": evaluation["evaluation"].get("images_per_second"),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(COMPARISON_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in COMPARISON_FIELDS})


def _format(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = (
        "experiment_id",
        "backbone_source",
        "trainable_backbone",
        "trainable_parameters",
        "best_epoch",
        "best_validation_total_loss",
        "map_50_95",
        "map_50",
        "map_75",
        "precision_at_0_5",
        "recall_at_0_5",
        "f1_at_0_5",
        "training_seconds",
    )
    lines = [
        "# Exercise 3.3 — Backbone experiment comparison",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(_format(row.get(column)) for column in columns) + " |"
        )
    lines.extend(
        [
            "",
            "All rows use the same dataset split, seed, detector architecture, "
            "training epochs and evaluation protocol.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _save_plots(rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    if not rows:
        return []
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [row["experiment_id"] for row in rows]
    x = np.arange(len(rows))
    created: list[str] = []

    figure, axis = plt.subplots(figsize=(9, 5))
    width = 0.24
    axis.bar(x - width, [row["map_50_95"] for row in rows], width, label="mAP 0.50:0.95")
    axis.bar(x, [row["map_50"] for row in rows], width, label="AP50")
    axis.bar(x + width, [row["map_75"] for row in rows], width, label="AP75")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Metric")
    axis.set_title("COCO-style validation metrics")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    path = output_dir / "comparison_map.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    created.append(str(path))

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(x - width, [row["precision_at_0_5"] for row in rows], width, label="Precision")
    axis.bar(x, [row["recall_at_0_5"] for row in rows], width, label="Recall")
    axis.bar(x + width, [row["f1_at_0_5"] for row in rows], width, label="F1")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Metric")
    axis.set_title("Fixed-threshold validation diagnostics")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    path = output_dir / "comparison_fixed_threshold.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    created.append(str(path))

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(x, [row["training_seconds"] / 60.0 for row in rows])
    axis.set_xticks(x, labels)
    axis.set_ylabel("Training minutes")
    axis.set_title("Training cost")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    path = output_dir / "comparison_training_time.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    created.append(str(path))
    return created


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [
        row
        for row in rows
        if row.get("map_50_95") is not None
        and math.isfinite(float(row["map_50_95"]))
    ]
    return max(valid, key=lambda row: float(row["map_50_95"])) if valid else None


def log_comparison_to_wandb(
    *,
    rows: list[dict[str, Any]],
    plot_paths: list[str],
    project: str,
    entity: str | None,
    group: str,
    mode: str,
    run_name: str,
    config: dict[str, Any],
) -> str:
    try:
        import wandb
    except ImportError as error:
        raise ImportError("W&B comparison logging requires wandb.") from error

    run = wandb.init(
        project=project,
        entity=entity,
        group=group,
        mode=mode,
        name=run_name,
        job_type="comparison",
        tags=["exercise-3-3", "comparison", "backbone-study"],
        config=config,
    )
    if run is None:
        raise RuntimeError("wandb.init() did not return a comparison run.")
    columns = list(COMPARISON_FIELDS)
    table = wandb.Table(
        columns=columns,
        data=[[row.get(column) for column in columns] for row in rows],
    )
    payload: dict[str, Any] = {"comparison/results": table}
    if plot_paths:
        payload["comparison/plots"] = [wandb.Image(path) for path in plot_paths]
    run.log(payload)

    best = _best_row(rows)
    if best is not None:
        run.summary["best_experiment_id"] = best["experiment_id"]
        run.summary["best_validation_map_50_95"] = best["map_50_95"]
        run.summary["best_validation_map_50"] = best["map_50"]
    run_id = str(run.id)
    run.finish(exit_code=0)
    return run_id


def write_comparison_artifacts(
    *,
    rows: list[dict[str, Any]],
    study_dir: Path,
) -> dict[str, Any]:
    study_dir.mkdir(parents=True, exist_ok=True)
    csv_path = study_dir / "comparison.csv"
    json_path = study_dir / "comparison.json"
    markdown_path = study_dir / "comparison.md"
    _write_csv(csv_path, rows)
    json_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_markdown(markdown_path, rows)
    plots = _save_plots(rows, study_dir)
    best = _best_row(rows)
    summary = {
        "completed_experiments": len(rows),
        "best_experiment_id": None if best is None else best["experiment_id"],
        "best_validation_map_50_95": None if best is None else best["map_50_95"],
        "artifacts": {
            "csv": str(csv_path),
            "json": str(json_path),
            "markdown": str(markdown_path),
            "plots": plots,
        },
    }
    (study_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary
