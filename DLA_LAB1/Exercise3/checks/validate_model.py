"""Build and audit the Step 9 Faster R-CNN baseline without a forward pass."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import torch
import torchvision

from Exercise3.models.faster_rcnn import (
    FasterRCNNBaselineConfig,
    build_faster_rcnn_baseline,
    summarize_faster_rcnn,
)
from Exercise3.paths import OUTPUT_DIR


DEFAULT_OUTPUT_PATH = OUTPUT_DIR / "step_9" / "model_validation.json"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct the COCO-pretrained Faster R-CNN ResNet-50-FPN baseline, "
            "replace its predictor for 44 classes, freeze the backbone and save "
            "a complete parameter audit."
        )
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cpu, cuda or cuda:<index>.",
    )
    parser.add_argument(
        "--weights",
        choices=("coco", "none"),
        default="coco",
        help=(
            "Use official COCO weights or random weights. The scientific "
            "baseline uses 'coco'; 'none' is only for offline structural tests."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Hide the Torchvision weight-download progress bar.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    device = torch.device(normalized)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        index = 0 if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {index} is invalid; found "
                f"{torch.cuda.device_count()} CUDA device(s)."
            )
    return device


def capture_cuda_memory(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {
            "available": False,
            "device_name": None,
            "allocated_bytes": None,
            "reserved_bytes": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }

    index = 0 if device.index is None else device.index
    return {
        "available": True,
        "device_index": index,
        "device_name": torch.cuda.get_device_name(index),
        "allocated_bytes": int(torch.cuda.memory_allocated(index)),
        "reserved_bytes": int(torch.cuda.memory_reserved(index)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
    }


def format_count(value: int) -> str:
    return f"{value:,}"


def format_megabytes(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value / (1024 ** 2):.2f} MiB"


def print_report(report: dict[str, Any]) -> None:
    parameters = report["parameters"]
    model_count = parameters["model"]
    device = report["runtime"]["device"]
    memory = report["runtime"]["cuda_memory_after_model_move"]

    print("\n=== Exercise 3.3 - Step 9: Faster R-CNN baseline ===")
    print(f"Architecture: {report['architecture']}")
    print(f"Initialization: {report['weights']}")
    print(
        "Predictor replacement: "
        f"{report['original_predictor_num_classes']} -> "
        f"{report['new_predictor_num_classes']} classes"
    )
    print(f"Background label: {report['background_label']}")
    print(
        "Foreground labels: "
        f"{report['foreground_label_range'][0]}.."
        f"{report['foreground_label_range'][1]}"
    )
    print(
        "Box-regression outputs: "
        f"{report['bbox_regression_outputs']} "
        "(4 values for each detector class)"
    )
    print(f"Device: {device}")
    if memory["available"]:
        print(f"CUDA device: {memory['device_name']}")
        print(
            "CUDA memory after model move: "
            f"allocated={format_megabytes(memory['allocated_bytes'])}, "
            f"reserved={format_megabytes(memory['reserved_bytes'])}"
        )

    print("\nParameter counts:")
    print(f"  total:     {format_count(model_count['total'])}")
    print(f"  trainable: {format_count(model_count['trainable'])}")
    print(f"  frozen:    {format_count(model_count['frozen'])}")
    print(
        "  trainable percentage: "
        f"{model_count['trainable_percentage']:.2f}%"
    )

    print("\nPer component:")
    for name, count in parameters["components"].items():
        print(
            f"  - {name}: total={format_count(count['total'])}, "
            f"trainable={format_count(count['trainable'])}, "
            f"frozen={format_count(count['frozen'])}"
        )

    print("\nFrozen components:")
    for name in parameters["frozen_components"]:
        print(f"  - {name}")

    print("\nTrainable components:")
    for name in parameters["trainable_components"]:
        print(f"  - {name}")

    print("\nModule modes prepared for training:")
    for name, value in report["model_modes"].items():
        print(f"  - {name}: {value}")

    print("\nModel construction and freezing checks: PASSED")


def main() -> None:
    arguments = parse_arguments()
    device = resolve_device(arguments.device)

    if device.type == "cuda":
        cuda_index = 0 if device.index is None else device.index

        # Imposta esplicitamente la GPU corrente. Alcune versioni di PyTorch
        # su Windows non accettano correttamente torch.device in
        # reset_peak_memory_stats().
        torch.cuda.set_device(cuda_index)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    memory_before = capture_cuda_memory(device)

    config = FasterRCNNBaselineConfig(
        weights=arguments.weights,
        seed=arguments.seed,
        progress=not arguments.no_progress,
    )
    model, construction_metadata = build_faster_rcnn_baseline(config)
    model.to(device)

    report = summarize_faster_rcnn(model, construction_metadata)
    report["runtime"] = {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_memory_before_model": memory_before,
        "cuda_memory_after_model_move": capture_cuda_memory(device),
        "forward_executed": False,
        "backward_executed": False,
        "optimizer_step_executed": False,
    }

    output_path = arguments.output_path.expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print_report(report)
    print(f"\nModel validation report saved to: {output_path}")


if __name__ == "__main__":
    main()
