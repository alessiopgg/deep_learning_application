"""Strict ResNet-50 GTSRB checkpoint inspection and detector transfer."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torchvision.models.detection import FasterRCNN

from Exercise3.paths import PROJECT_ROOT


TRANSFER_PREFIXES = (
    "conv1.",
    "bn1.",
    "layer1.",
    "layer2.",
    "layer3.",
    "layer4.",
)


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a path relative to the DLA_LAB1 project root."""

    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Return a reproducible SHA-256 checksum for one tensor."""

    contiguous = tensor.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def load_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    """Load a classification checkpoint on CPU and validate its root type."""

    resolved = resolve_project_path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"GTSRB checkpoint not found: {resolved}")

    try:
        payload = torch.load(
            resolved,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(resolved, map_location="cpu")

    if not isinstance(payload, dict):
        raise TypeError("GTSRB checkpoint root must be a dictionary.")
    return payload


def extract_state_dict(
    checkpoint: Mapping[str, Any],
) -> Mapping[str, torch.Tensor]:
    """Read the model state dictionary from supported checkpoint schemas."""

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        state_dict = checkpoint.get("state_dict")

    if not isinstance(state_dict, Mapping):
        raise KeyError(
            "GTSRB checkpoint must contain model_state_dict or state_dict."
        )
    if not all(isinstance(key, str) for key in state_dict):
        raise TypeError("Checkpoint state-dict keys must be strings.")

    return state_dict  # type: ignore[return-value]


def checkpoint_identity(
    checkpoint: Mapping[str, Any],
) -> tuple[str | None, str | None, str | None]:
    """Extract model, fine-tuning strategy and classifier type metadata."""

    config = checkpoint.get("config", {})
    model_name: str | None = None
    strategy: str | None = None
    classifier_type: str | None = None

    if isinstance(config, Mapping):
        model_section = config.get("model")
        if isinstance(model_section, Mapping):
            raw_model = model_section.get("name")
            raw_strategy = model_section.get("fine_tuning_strategy")
            raw_classifier = model_section.get("classifier_type")
            model_name = None if raw_model is None else str(raw_model)
            strategy = None if raw_strategy is None else str(raw_strategy)
            classifier_type = (
                None if raw_classifier is None else str(raw_classifier)
            )
        elif isinstance(model_section, str):
            model_name = model_section

        if strategy is None and config.get("strategy") is not None:
            strategy = str(config.get("strategy"))
        if (
            classifier_type is None
            and config.get("classifier_type") is not None
        ):
            classifier_type = str(config.get("classifier_type"))

    return model_name, strategy, classifier_type


def normalize_source_key(key: str) -> str:
    """Normalize common wrappers around ResNet state-dict keys."""

    normalized = key
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "model."):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                changed = True

    for prefix in ("backbone.body.", "backbone.", "body."):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break

    return normalized


def _normalized_tensor_state(
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    normalized_source: dict[str, torch.Tensor] = {}
    original_keys: dict[str, str] = {}

    for source_key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue

        normalized = normalize_source_key(source_key)
        if normalized in normalized_source:
            raise ValueError(
                "Multiple checkpoint tensors map to the same normalized key "
                f"{normalized!r}: {original_keys[normalized]!r}, "
                f"{source_key!r}."
            )

        normalized_source[normalized] = value.detach().cpu()
        original_keys[normalized] = source_key

    return normalized_source, original_keys


def inspect_gtsrb_checkpoint(
    checkpoint_path: str | Path,
    *,
    required_model: str = "resnet50",
    required_strategy: str | None = "full",
    required_classifier_type: str | None = None,
) -> dict[str, Any]:
    """Validate checkpoint identity and expose transfer-relevant metadata."""

    resolved = resolve_project_path(checkpoint_path).resolve()
    checkpoint = load_checkpoint_payload(resolved)
    state_dict = extract_state_dict(checkpoint)
    model_name, strategy, classifier_type = checkpoint_identity(checkpoint)

    actual_model = (model_name or "").strip().lower()
    expected_model = required_model.strip().lower()
    if actual_model != expected_model:
        raise ValueError(
            "The GTSRB checkpoint has the wrong model: "
            f"expected {expected_model!r}, found {model_name!r}."
        )

    if required_strategy is not None:
        actual_strategy = (
            None if strategy is None else strategy.strip().lower()
        )
        expected_strategy = required_strategy.strip().lower()
        if actual_strategy != expected_strategy:
            raise ValueError(
                "The GTSRB checkpoint has the wrong fine-tuning strategy: "
                f"expected {expected_strategy!r}, found {strategy!r}."
            )

    if required_classifier_type is not None:
        actual_classifier = (
            None
            if classifier_type is None
            else classifier_type.strip().lower()
        )
        expected_classifier = required_classifier_type.strip().lower()
        if actual_classifier != expected_classifier:
            raise ValueError(
                "The GTSRB checkpoint has the wrong classifier type: "
                f"expected {expected_classifier!r}, "
                f"found {classifier_type!r}."
            )

    normalized_source, original_keys = _normalized_tensor_state(state_dict)
    transfer_keys = sorted(
        key
        for key in normalized_source
        if key.startswith(TRANSFER_PREFIXES)
    )
    missing_stages = [
        stage.rstrip(".")
        for stage in TRANSFER_PREFIXES
        if not any(key.startswith(stage) for key in transfer_keys)
    ]
    if missing_stages:
        raise KeyError(
            "The GTSRB checkpoint does not contain all ResNet-50 stages: "
            + ", ".join(missing_stages)
        )

    monitor = checkpoint.get("monitor")
    monitored_value = checkpoint.get("monitored_value")
    validation_loss = checkpoint.get("best_validation_loss")
    if validation_loss is None and monitor in {
        "validation_loss",
        "validation_total_loss",
    }:
        validation_loss = monitored_value

    return {
        "checkpoint_path": str(resolved),
        "checkpoint_model": model_name,
        "checkpoint_strategy": strategy,
        "checkpoint_classifier_type": classifier_type,
        "checkpoint_epoch": checkpoint.get(
            "epoch",
            checkpoint.get("best_epoch"),
        ),
        "checkpoint_monitor": monitor,
        "checkpoint_monitored_value": monitored_value,
        "checkpoint_validation_loss": validation_loss,
        "state_tensor_count": len(normalized_source),
        "transfer_tensor_count": len(transfer_keys),
        "transfer_stages": [
            "conv1",
            "bn1",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
        ],
        "excluded_modules": ["avgpool", "fc"],
        "classifier_tensor_keys": sorted(
            key for key in normalized_source if key.startswith("fc.")
        ),
        "source_key_examples": {
            key: original_keys[key] for key in transfer_keys[:10]
        },
    }


def load_gtsrb_backbone(
    model: FasterRCNN,
    *,
    checkpoint_path: str | Path,
    required_strategy: str | None,
) -> dict[str, Any]:
    """Transfer only conv1, bn1 and layer1-layer4 into the detector."""

    resolved = resolve_project_path(checkpoint_path).resolve()
    checkpoint = load_checkpoint_payload(resolved)
    state_dict = extract_state_dict(checkpoint)
    identity = inspect_gtsrb_checkpoint(
        resolved,
        required_model="resnet50",
        required_strategy=required_strategy,
    )
    normalized_source, original_keys = _normalized_tensor_state(state_dict)

    body = model.backbone.body
    target_state = body.state_dict()
    target_keys = [
        key for key in target_state if key.startswith(TRANSFER_PREFIXES)
    ]

    if set(target_keys) != set(target_state):
        unexpected_target = sorted(set(target_state) - set(target_keys))
        raise ValueError(
            "The Faster R-CNN ResNet body exposes unexpected tensors outside "
            "conv1/bn1/layer1..4: "
            f"{unexpected_target[:10]}"
        )

    missing = [key for key in target_keys if key not in normalized_source]
    if missing:
        raise KeyError(
            "GTSRB checkpoint is missing backbone tensors: "
            + ", ".join(missing[:20])
        )

    shape_mismatches: list[dict[str, Any]] = []
    for key in target_keys:
        source = normalized_source[key]
        target = target_state[key]
        if tuple(source.shape) != tuple(target.shape):
            shape_mismatches.append(
                {
                    "key": key,
                    "source_shape": list(source.shape),
                    "target_shape": list(target.shape),
                }
            )
    if shape_mismatches:
        raise ValueError(
            f"GTSRB backbone shape mismatches: {shape_mismatches[:5]}"
        )

    representative_name = "conv1.weight"
    before_checksum = tensor_sha256(target_state[representative_name])
    copied = {
        key: normalized_source[key].to(dtype=target_state[key].dtype)
        for key in target_keys
    }

    with torch.no_grad():
        body.load_state_dict(copied, strict=True)

    loaded_state = body.state_dict()
    verification_failures = [
        key
        for key in target_keys
        if not torch.equal(loaded_state[key].cpu(), copied[key].cpu())
    ]
    after_checksum = tensor_sha256(loaded_state[representative_name])

    return {
        **identity,
        "target_modules": [
            "conv1",
            "bn1",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
        ],
        "excluded_modules": ["avgpool", "fc"],
        "target_tensor_count": len(target_keys),
        "loaded_tensor_count": len(copied),
        "all_target_tensors_loaded": len(copied) == len(target_keys),
        "shape_mismatches": shape_mismatches,
        "verification_failures": verification_failures,
        "exact_post_load_verification": not verification_failures,
        "representative_tensor": "backbone.body.conv1.weight",
        "representative_sha256_before_coco": before_checksum,
        "representative_sha256_after_gtsrb": after_checksum,
        "representative_tensor_changed_from_coco": (
            before_checksum != after_checksum
        ),
        "source_key_examples": {
            key: original_keys[key] for key in target_keys[:10]
        },
    }
