"""CSV, plots and Markdown summaries for detector evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_training_plots(run_dir: Path, output_dir: Path) -> list[str]:
    history_path = run_dir / "history.csv"
    if not history_path.is_file():
        return []
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("matplotlib is required for training plots.") from error

    with history_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return []

    epochs = [int(row["epoch"]) for row in rows]
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, [float(row["train_total_loss"]) for row in rows], marker="o", label="train")
    axis.plot(epochs, [float(row["validation_total_loss"]) for row in rows], marker="o", label="validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Total loss")
    axis.set_title("Faster R-CNN total loss")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    path = output_dir / "loss_total.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    created.append(str(path))

    for prefix, title in (("train", "Training loss components"), ("validation", "Validation loss components")):
        figure, axis = plt.subplots(figsize=(9, 5))
        for key, label in (
            ("loss_classifier", "classifier"),
            ("loss_box_reg", "box regression"),
            ("loss_objectness", "RPN objectness"),
            ("loss_rpn_box_reg", "RPN box regression"),
        ):
            axis.plot(
                epochs,
                [float(row[f"{prefix}_{key}"]) for row in rows],
                marker="o",
                label=label,
            )
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        path = output_dir / f"loss_components_{prefix}.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        created.append(str(path))

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(
        epochs,
        [float(row["learning_rate"]) for row in rows],
        marker="o",
        label="detector heads",
    )
    backbone_values = [row.get("backbone_learning_rate", "") for row in rows]
    if any(value not in {"", None, "None"} for value in backbone_values):
        axis.plot(
            epochs,
            [
                float(value) if value not in {"", None, "None"} else float("nan")
                for value in backbone_values
            ],
            marker="o",
            label="trainable backbone",
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Learning rate")
    axis.set_title("Learning-rate schedule")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    path = output_dir / "learning_rate.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    created.append(str(path))
    return created


def format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def build_markdown_summary(report: dict[str, Any]) -> str:
    standard = report["metrics"]["coco_style"]
    fixed = report["metrics"]["fixed_threshold"]
    checkpoint = report["checkpoint"]
    lines = [
        "# Exercise 3.3 — Detector evaluation",
        "",
        f"- Split: `{report['evaluation']['split']}`",
        f"- Checkpoint: `{checkpoint['path']}`",
        f"- Checkpoint epoch: `{checkpoint['epoch']}`",
        f"- COCO backend: `{report['evaluation']['backend']}`",
        f"- Images evaluated: `{report['dataset']['images_evaluated']}`",
        "",
        "## COCO-style metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| mAP@[0.50:0.95] | {format_metric(standard['map'])} |",
        f"| AP50 | {format_metric(standard['map_50'])} |",
        f"| AP75 | {format_metric(standard['map_75'])} |",
        f"| AP small | {format_metric(standard['map_small'])} |",
        f"| AP medium | {format_metric(standard['map_medium'])} |",
        f"| AP large | {format_metric(standard['map_large'])} |",
        f"| AR@100 | {format_metric(standard['mar_100'])} |",
        "",
        "## Fixed-threshold diagnostics",
        "",
        f"Score threshold: `{report['evaluation']['fixed_score_threshold']}`  ",
        f"IoU threshold: `{report['evaluation']['fixed_iou_threshold']}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Precision | {fixed['precision']:.6f} |",
        f"| Recall | {fixed['recall']:.6f} |",
        f"| F1 | {fixed['f1']:.6f} |",
        f"| True positives | {fixed['true_positives']} |",
        f"| False positives | {fixed['false_positives']} |",
        f"| False negatives | {fixed['false_negatives']} |",
        f"| Empty-image FP rate | {format_metric(fixed['empty_target_false_positive_rate'])} |",
        "",
        "## Protocol",
        "",
        f"- Scientific full evaluation: `{report['evaluation']['scientific_full_evaluation']}`",
        f"- Test split used: `{report['evaluation']['split'] == 'test'}`",
        "- Model selection must use validation results; final test evaluation should be run only after configurations are fixed.",
        "",
    ]
    return "\n".join(lines)
