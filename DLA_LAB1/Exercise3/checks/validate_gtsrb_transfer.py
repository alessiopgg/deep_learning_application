"""Strict preflight validation for the ResNet-50 GTSRB checkpoint transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from Exercise3.models.faster_rcnn import (
    FasterRCNNBaselineConfig,
    build_faster_rcnn_baseline,
    summarize_faster_rcnn,
)
from Exercise3.paths import OUTPUT_DIR, PROJECT_ROOT


DEFAULT_OUTPUT = OUTPUT_DIR / "step_14" / "gtsrb_transfer_validation.json"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate exact transfer of a GTSRB ResNet-50 checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--required-strategy", type=str, default="full")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else PROJECT_ROOT / expanded


def main() -> None:
    arguments = parse_arguments()
    checkpoint = resolve_project_path(arguments.checkpoint).resolve()
    output = arguments.output.expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output

    model, metadata = build_faster_rcnn_baseline(
        FasterRCNNBaselineConfig(
            weights="coco",
            backbone_source="gtsrb",
            gtsrb_checkpoint=str(checkpoint),
            required_gtsrb_strategy=arguments.required_strategy,
            trainable_backbone="frozen",
            freeze_backbone=True,
            progress=not arguments.no_progress,
        )
    )
    audit = summarize_faster_rcnn(model, metadata)
    transfer = metadata["gtsrb_transfer"]
    report = {
        "checkpoint": str(checkpoint),
        "transfer": transfer,
        "model_audit": audit,
        "all_checks_passed": bool(
            transfer
            and transfer["all_target_tensors_loaded"]
            and transfer["exact_post_load_verification"]
            and transfer["representative_tensor_changed_from_coco"]
        ),
    }
    if not report["all_checks_passed"]:
        raise RuntimeError("GTSRB transfer preflight did not pass all checks.")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== Exercise 3.3 - GTSRB transfer preflight ===")
    print(f"Checkpoint: {checkpoint}")
    print(f"Checkpoint model: {transfer['checkpoint_model']}")
    print(f"Checkpoint strategy: {transfer['checkpoint_strategy']}")
    print(
        "Transferred tensors: "
        f"{transfer['loaded_tensor_count']} / {transfer['target_tensor_count']}"
    )
    print(
        "Representative checksum changed from COCO: "
        f"{transfer['representative_tensor_changed_from_coco']}"
    )
    print("Exact post-load verification: True")
    print("All GTSRB transfer checks: PASSED")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
