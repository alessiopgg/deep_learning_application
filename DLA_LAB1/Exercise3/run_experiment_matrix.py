"""Run A/B/C/D sequentially, evaluate each run and compare all results."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import subprocess
import sys

import torch
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from Exercise3.experiments.comparison import (
    build_comparison_row,
    log_comparison_to_wandb,
    write_comparison_artifacts,
)
from Exercise3.experiments.matrix import (
    ExperimentSpec,
    StudyConfig,
    load_study_config,
    resolve_project_path,
)
from Exercise3.training.configuration import (
    load_training_config,
    resolve_output_root,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete Exercise 3.3 backbone experiment matrix: "
            "training, validation evaluation, W&B tracking and comparison."
        )
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("Exercise3/configs/experiment_matrix.yaml"),
    )
    parser.add_argument(
        "--gtsrb-checkpoint",
        type=str,
        default=None,
        help="Checkpoint path or auto to select the best ResNet-50 full run.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--log-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Upload each best_model.pt to W&B (large total upload).",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Optional subset of experiment IDs, e.g. A B C D.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-study", type=Path, default=None)
    parser.add_argument(
        "--evaluate-test-all",
        action="store_true",
        help="Explicitly evaluate test for every run. Avoid during model selection.",
    )
    return parser.parse_args()


def _sanitize(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    normalized = normalized.strip("-._")
    if not normalized:
        raise ValueError("Study or experiment name became empty after sanitization.")
    return normalized


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _null_or_string(value: str | None) -> str:
    return "null" if value is None else value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _run_tee(
    command: list[str],
    *,
    log_path: Path,
    dry_run: bool,
) -> None:
    printable = subprocess.list2cmdline(command)
    print("\n$ " + printable, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + printable + "\n")
        log.flush()
        if dry_run:
            return
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _new_run_directory(output_root: Path, before: set[Path]) -> Path:
    after = {
        path.resolve()
        for path in output_root.iterdir()
        if path.is_dir()
    }
    new = sorted(
        after - before,
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not new:
        raise RuntimeError("Training finished but no new run directory was found.")
    return new[0]


def _selected_experiments(
    config: StudyConfig,
    selected_ids: list[str] | None,
) -> list[ExperimentSpec]:
    enabled = config.enabled_experiments()
    if selected_ids is None:
        return enabled
    requested = set(selected_ids)
    known = {experiment.id for experiment in config.experiments}
    unknown = requested - known
    if unknown:
        raise ValueError(f"Unknown experiment IDs: {sorted(unknown)}")
    selected = [experiment for experiment in enabled if experiment.id in requested]
    if not selected:
        raise ValueError("No enabled experiment remains after filtering.")
    return selected


def _apply_cli_overrides(config: StudyConfig, args: argparse.Namespace) -> None:
    if args.gtsrb_checkpoint is not None:
        config.gtsrb_checkpoint = str(args.gtsrb_checkpoint)
    if args.device is not None:
        config.device = args.device
    if args.num_workers is not None:
        config.num_workers = args.num_workers
        config.persistent_workers = args.num_workers > 0
    if args.wandb_project is not None:
        config.wandb_project = args.wandb_project
    if args.wandb_entity is not None:
        config.wandb_entity = args.wandb_entity
    if args.wandb_group is not None:
        config.wandb_group = args.wandb_group
    if args.wandb is not None:
        config.wandb_enabled = args.wandb
    if args.log_checkpoints is not None:
        config.log_best_checkpoints = args.log_checkpoints
    if args.continue_on_error:
        config.continue_on_error = True
    if config.num_workers < 0:
        raise ValueError("num_workers cannot be negative.")



def _distribution_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _validate_runtime_dependencies(
    *,
    config: StudyConfig,
    study_dir: Path,
) -> None:
    required_modules = {
        "datasets": "datasets",
        "matplotlib": "matplotlib",
        "numpy": "numpy",
        "omegaconf": "omegaconf",
        "PIL": "Pillow",
        "torch": "torch",
        "torchvision": "torchvision",
        "torchmetrics": "torchmetrics",
    }
    missing = [
        distribution
        for module, distribution in required_modules.items()
        if importlib.util.find_spec(module) is None
    ]
    if config.wandb_enabled and config.wandb_mode != "disabled":
        if importlib.util.find_spec("wandb") is None:
            missing.append("wandb")

    coco_backends = {
        "pycocotools": importlib.util.find_spec("pycocotools") is not None,
        "faster-coco-eval": (
            importlib.util.find_spec("faster_coco_eval") is not None
        ),
    }
    if not any(coco_backends.values()):
        missing.append("pycocotools or faster-coco-eval")

    versions = {
        distribution: _distribution_version(distribution)
        for distribution in sorted(set(required_modules.values()))
    }
    if config.wandb_enabled and config.wandb_mode != "disabled":
        versions["wandb"] = _distribution_version("wandb")
    versions.update(
        {
            "pycocotools": _distribution_version("pycocotools"),
            "faster-coco-eval": _distribution_version("faster-coco-eval"),
        }
    )

    datasets_version = versions.get("datasets")
    datasets_compatible = datasets_version == "3.6.0"
    report = {
        "missing_dependencies": sorted(set(missing)),
        "versions": versions,
        "coco_backends": coco_backends,
        "datasets_required_version": "3.6.0",
        "datasets_version_compatible": datasets_compatible,
        "wandb_required": (
            config.wandb_enabled and config.wandb_mode != "disabled"
        ),
    }
    _atomic_json(study_dir / "dependency_preflight.json", report)

    if missing:
        raise RuntimeError(
            "Missing experiment-matrix dependencies: "
            + ", ".join(sorted(set(missing)))
        )
    if not datasets_compatible:
        raise RuntimeError(
            "This dataset loader requires datasets==3.6.0; found "
            f"{datasets_version!r}."
        )
    print("\nDependency preflight: PASSED")
    print(f"  datasets: {datasets_version}")
    print(f"  torchmetrics: {versions.get('torchmetrics')}")
    print(
        "  COCO backend: "
        + (
            "pycocotools"
            if coco_backends["pycocotools"]
            else "faster-coco-eval"
        )
    )
    if report["wandb_required"]:
        print(f"  wandb: {versions.get('wandb')}")


def _scientific_study_signature(config: StudyConfig) -> dict[str, Any]:
    """Fields that must not change when resuming an existing study."""
    payload = {
        "name": config.name,
        "training_config": config.training_config,
        "evaluation_config": config.evaluation_config,
        "required_gtsrb_strategy": config.required_gtsrb_strategy,
        "gtsrb_checkpoint": config.gtsrb_checkpoint,
        "common_training_overrides": list(config.common_training_overrides),
        "common_evaluation_overrides": list(config.common_evaluation_overrides),
        "experiments": [asdict(item) for item in config.experiments],
    }
    # Normalize tuples to the JSON representation stored in study_manifest.json.
    return json.loads(json.dumps(payload))


def _prepare_study_directory(
    config: StudyConfig,
    args: argparse.Namespace,
) -> tuple[str, Path, dict[str, Any]]:
    if args.resume_study is not None:
        study_dir = resolve_project_path(args.resume_study).resolve()
        manifest_path = study_dir / "study_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Study manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored_config = manifest.get("config")
        if not isinstance(stored_config, dict):
            raise ValueError("Study manifest does not contain the original config.")
        stored_signature = {
            "name": stored_config.get("name"),
            "training_config": stored_config.get("training_config"),
            "evaluation_config": stored_config.get("evaluation_config"),
            "required_gtsrb_strategy": stored_config.get("required_gtsrb_strategy"),
            "gtsrb_checkpoint": stored_config.get("gtsrb_checkpoint"),
            "common_training_overrides": stored_config.get("common_training_overrides", []),
            "common_evaluation_overrides": stored_config.get("common_evaluation_overrides", []),
            "experiments": stored_config.get("experiments", []),
        }
        current_signature = _scientific_study_signature(config)
        if stored_signature != current_signature:
            raise ValueError(
                "The matrix scientific configuration changed since this study "
                "was created. Start a new study instead of resuming it."
            )
        return str(manifest["study_id"]), study_dir, manifest

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_id = f"{timestamp}_{_sanitize(config.name)}"
    output_root = resolve_project_path(config.output_dir)
    study_dir = output_root / study_id
    study_dir.mkdir(parents=True, exist_ok=False)
    group = config.wandb_group or study_id
    manifest = {
        "study_id": study_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "study_dir": str(study_dir),
        "wandb_group": group,
        "config": asdict(config),
        "experiments": {},
        "comparison": None,
    }
    _atomic_json(study_dir / "study_manifest.json", manifest)
    return study_id, study_dir, manifest


def _classification_checkpoint_metadata(
    path: Path,
    *,
    required_strategy: str | None,
) -> dict[str, Any] | None:
    try:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(checkpoint, Mapping):
        return None

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping):
        return None

    config = checkpoint.get("config", {})
    model_name = None
    strategy = None
    classifier_type = None
    if isinstance(config, Mapping):
        model_section = config.get("model")
        if isinstance(model_section, Mapping):
            model_name = model_section.get("name")
            strategy = model_section.get("fine_tuning_strategy")
            classifier_type = model_section.get("classifier_type")
        elif isinstance(model_section, str):
            model_name = model_section
        strategy = strategy or config.get("strategy")
        classifier_type = classifier_type or config.get("classifier_type")

    normalized_model = str(model_name).strip().lower()
    normalized_strategy = None if strategy is None else str(strategy).strip().lower()
    if normalized_model != "resnet50":
        return None
    if (
        required_strategy is not None
        and normalized_strategy != required_strategy.strip().lower()
    ):
        return None

    validation_loss = checkpoint.get("best_validation_loss")
    if validation_loss is None:
        monitor = checkpoint.get("monitor")
        if monitor in {"validation_loss", "validation_total_loss"}:
            validation_loss = checkpoint.get("monitored_value")
    # Do not use a generic best_metric fallback: it may be accuracy or F1.

    normalized_loss = None
    if validation_loss is not None:
        try:
            candidate_loss = float(validation_loss)
        except (TypeError, ValueError):
            candidate_loss = math.nan
        if math.isfinite(candidate_loss):
            normalized_loss = candidate_loss

    return {
        "path": str(path.resolve()),
        "model": str(model_name),
        "strategy": None if strategy is None else str(strategy),
        "classifier_type": (
            None if classifier_type is None else str(classifier_type)
        ),
        "validation_loss": normalized_loss,
        "epoch": checkpoint.get("epoch", checkpoint.get("best_epoch")),
        "modified_timestamp": path.stat().st_mtime,
    }


def _discover_gtsrb_checkpoint(
    study_dir: Path,
    *,
    required_strategy: str | None,
) -> Path:
    roots = [
        resolve_project_path("Exercise2/outputs"),
        resolve_project_path("Exercise1/outputs"),
    ]
    candidates: list[dict[str, Any]] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("best_model.pt"):
            metadata = _classification_checkpoint_metadata(
                path,
                required_strategy=required_strategy,
            )
            if metadata is not None:
                candidates.append(metadata)
    if not candidates:
        strategy_text = required_strategy or "any"
        raise FileNotFoundError(
            "Automatic checkpoint discovery found no compatible "
            f"ResNet-50/{strategy_text} best_model.pt under "
            "Exercise1/outputs or Exercise2/outputs."
        )

    scored = [
        candidate
        for candidate in candidates
        if candidate["validation_loss"] is not None
    ]
    if not scored:
        report = {
            "selection_policy": (
                "ResNet-50 checkpoint with the required fine-tuning strategy "
                "and minimum finite stored validation loss"
            ),
            "required_strategy": required_strategy,
            "selected": None,
            "candidates": candidates,
            "error": "No compatible candidate stores a finite validation loss.",
        }
        (study_dir / "gtsrb_checkpoint_selection.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        raise ValueError(
            "Compatible ResNet-50 checkpoints were found, but none stores a "
            "finite validation loss. Pass an explicit --gtsrb-checkpoint path "
            "instead of auto."
        )

    ranked = sorted(
        scored,
        key=lambda item: (
            float(item["validation_loss"]),
            -float(item["modified_timestamp"]),
            item["path"],
        ),
    )
    selected = ranked[0]
    report = {
        "selection_policy": (
            "ResNet-50 checkpoint with the required fine-tuning strategy; "
            "minimum finite stored validation loss; newest file breaks ties"
        ),
        "required_strategy": required_strategy,
        "selected": selected,
        "ranked_candidates": ranked,
        "compatible_candidates_without_validation_loss": [
            candidate
            for candidate in candidates
            if candidate["validation_loss"] is None
        ],
    }
    (study_dir / "gtsrb_checkpoint_selection.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\nAutomatically selected GTSRB checkpoint:")
    print(f"  {selected['path']}")
    print(f"  validation loss: {selected['validation_loss']}")
    print(f"  strategy: {selected['strategy']}")
    return Path(selected["path"])


def _resolve_gtsrb_checkpoint(
    config: StudyConfig,
    experiments: list[ExperimentSpec],
    *,
    study_dir: Path,
    dry_run: bool,
) -> Path | None:
    needs_gtsrb = any(
        experiment.backbone_source == "gtsrb" for experiment in experiments
    )
    if not needs_gtsrb:
        return None
    if not config.gtsrb_checkpoint:
        raise ValueError(
            "The selected matrix includes GTSRB runs. Supply "
            "--gtsrb-checkpoint auto or a best_model.pt path."
        )
    raw = str(config.gtsrb_checkpoint)
    if raw.strip().lower() == "auto":
        return _discover_gtsrb_checkpoint(
            study_dir,
            required_strategy=config.required_gtsrb_strategy,
        )
    path = resolve_project_path(raw).resolve()
    if not dry_run and not path.is_file():
        raise FileNotFoundError(f"GTSRB checkpoint not found: {path}")
    return path

def _training_overrides(
    *,
    config: StudyConfig,
    experiment: ExperimentSpec,
    study_id: str,
    group: str,
    gtsrb_checkpoint: Path | None,
) -> list[str]:
    freeze = experiment.trainable_backbone == "frozen"
    checkpoint_value = (
        None
        if experiment.backbone_source == "coco"
        else str(gtsrb_checkpoint)
    )
    overrides = [
        *config.common_training_overrides,
        f"experiment.device={config.device}",
        f"experiment.run_name={study_id}-{experiment.id}-{_sanitize(experiment.name)}",
        f"data.num_workers={config.num_workers}",
        f"data.persistent_workers={_bool(config.persistent_workers)}",
        f"tracking.use_wandb={_bool(config.wandb_enabled)}",
        f"tracking.project={config.wandb_project}",
        f"tracking.group={group}",
        f"tracking.mode={config.wandb_mode}",
        f"tracking.log_best_checkpoint={_bool(config.log_best_checkpoints)}",
        f"model.backbone_source={experiment.backbone_source}",
        f"model.gtsrb_checkpoint={_null_or_string(checkpoint_value)}",
        f"model.required_gtsrb_strategy={_null_or_string(config.required_gtsrb_strategy)}",
        f"model.trainable_backbone={experiment.trainable_backbone}",
        f"model.freeze_backbone={_bool(freeze)}",
        f"optimizer.learning_rate={experiment.detector_learning_rate}",
        f"optimizer.backbone_learning_rate={experiment.backbone_learning_rate}",
        *experiment.extra_training_overrides,
    ]
    if config.wandb_entity is not None:
        overrides.append(f"tracking.entity={config.wandb_entity}")
    return overrides


def _evaluation_overrides(
    *,
    config: StudyConfig,
    experiment: ExperimentSpec,
) -> list[str]:
    return [
        *config.common_evaluation_overrides,
        f"runtime.device={config.device}",
        f"runtime.num_workers={config.num_workers}",
        f"runtime.persistent_workers={_bool(config.persistent_workers)}",
        f"tracking.enabled={_bool(config.wandb_enabled)}",
        "tracking.resume_training_run=true",
        f"tracking.log_evaluation_artifact={_bool(config.log_evaluation_artifacts)}",
        *experiment.extra_evaluation_overrides,
    ]


def main() -> None:
    args = parse_arguments()
    config = load_study_config(args.matrix)
    _apply_cli_overrides(config, args)
    experiments = _selected_experiments(config, args.experiments)
    study_id, study_dir, manifest = _prepare_study_directory(config, args)
    gtsrb_checkpoint = _resolve_gtsrb_checkpoint(
        config,
        experiments,
        study_dir=study_dir,
        dry_run=args.dry_run,
    )

    group = str(manifest["wandb_group"])
    (study_dir / "resolved_matrix.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    training_config_path = resolve_project_path(config.training_config).resolve()
    evaluation_config_path = resolve_project_path(config.evaluation_config).resolve()
    base_training = load_training_config(
        training_config_path,
        config.common_training_overrides,
    )
    training_output_root = resolve_output_root(base_training)
    training_output_root.mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        _validate_runtime_dependencies(config=config, study_dir=study_dir)

    if gtsrb_checkpoint is not None:
        preflight_command = [
            sys.executable,
            "-m",
            "Exercise3.checks.validate_gtsrb_transfer",
            "--checkpoint",
            str(gtsrb_checkpoint),
            "--output",
            str(study_dir / "gtsrb_transfer_validation.json"),
        ]
        if config.required_gtsrb_strategy is not None:
            preflight_command.extend(
                ["--required-strategy", config.required_gtsrb_strategy]
            )
        _run_tee(
            preflight_command,
            log_path=study_dir / "preflight.log",
            dry_run=args.dry_run,
        )

    if args.preflight_only:
        print("\nPreflight completed. No training was started.")
        return

    completed_rows: list[dict[str, Any]] = []
    for experiment in experiments:
        previous = manifest["experiments"].get(experiment.id, {})
        previous_run_dir = previous.get("run_dir")
        if previous.get("status") == "completed" and previous_run_dir:
            run_dir = Path(previous_run_dir)
            print(f"\nSkipping completed experiment {experiment.id}: {run_dir}")
            completed_rows.append(
                build_comparison_row(
                    experiment_id=experiment.id,
                    experiment_name=experiment.name,
                    run_dir=run_dir,
                )
            )
            continue

        reusable_run_dir: Path | None = None
        if previous_run_dir:
            candidate_run_dir = Path(previous_run_dir).expanduser().resolve()
            if (candidate_run_dir / "best_model.pt").is_file():
                reusable_run_dir = candidate_run_dir

        record = {
            "id": experiment.id,
            "name": experiment.name,
            "status": "running",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "run_dir": (
                None if reusable_run_dir is None else str(reusable_run_dir)
            ),
            "error": None,
        }
        manifest["experiments"][experiment.id] = record
        _atomic_json(study_dir / "study_manifest.json", manifest)

        try:
            if reusable_run_dir is not None:
                run_dir = reusable_run_dir
                checkpoint = run_dir / "best_model.pt"
                print(
                    f"\nReusing completed training for experiment "
                    f"{experiment.id}: {run_dir}"
                )
            else:
                before = {
                    path.resolve()
                    for path in training_output_root.iterdir()
                    if path.is_dir()
                }
                train_overrides = _training_overrides(
                    config=config,
                    experiment=experiment,
                    study_id=study_id,
                    group=group,
                    gtsrb_checkpoint=gtsrb_checkpoint,
                )
                training_command = [
                    sys.executable,
                    "-m",
                    "Exercise3.train_baseline",
                    "--config",
                    str(training_config_path),
                    *train_overrides,
                ]
                _run_tee(
                    training_command,
                    log_path=study_dir / "logs" / f"{experiment.id}_training.log",
                    dry_run=args.dry_run,
                )
                if args.dry_run:
                    continue

                run_dir = _new_run_directory(training_output_root, before)
                record["run_dir"] = str(run_dir)
                checkpoint = run_dir / "best_model.pt"
                if not checkpoint.is_file():
                    raise FileNotFoundError(
                        f"Best checkpoint not found: {checkpoint}"
                    )

            record["status"] = "evaluating"
            _atomic_json(study_dir / "study_manifest.json", manifest)
            evaluation_command = [
                sys.executable,
                "-m",
                "Exercise3.evaluate_detector",
                "--config",
                str(evaluation_config_path),
                "--checkpoint",
                str(checkpoint),
                "--split",
                "validation",
                * _evaluation_overrides(config=config, experiment=experiment),
            ]
            _run_tee(
                evaluation_command,
                log_path=study_dir / "logs" / f"{experiment.id}_validation.log",
                dry_run=args.dry_run,
            )
            if args.dry_run:
                continue

            if args.evaluate_test_all:
                test_command = [
                    sys.executable,
                    "-m",
                    "Exercise3.evaluate_detector",
                    "--config",
                    str(evaluation_config_path),
                    "--checkpoint",
                    str(checkpoint),
                    "--split",
                    "test",
                    "--allow-test",
                    *_evaluation_overrides(config=config, experiment=experiment),
                ]
                _run_tee(
                    test_command,
                    log_path=study_dir / "logs" / f"{experiment.id}_test.log",
                    dry_run=False,
                )

            row = build_comparison_row(
                experiment_id=experiment.id,
                experiment_name=experiment.name,
                run_dir=run_dir,
            )
            completed_rows.append(row)
            record.update(
                {
                    "status": "completed",
                    "completed_at": datetime.now().isoformat(timespec="seconds"),
                    "comparison_row": row,
                }
            )
        except BaseException as error:
            record.update(
                {
                    "status": "failed",
                    "failed_at": datetime.now().isoformat(timespec="seconds"),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            _atomic_json(study_dir / "study_manifest.json", manifest)
            print(f"\nExperiment {experiment.id} FAILED: {error}")
            if not config.continue_on_error:
                raise
        finally:
            manifest["experiments"][experiment.id] = record
            _atomic_json(study_dir / "study_manifest.json", manifest)

    if args.dry_run:
        print("\nDry run completed. No training or evaluation was executed.")
        return
    if not completed_rows:
        raise RuntimeError("No experiment completed; comparison cannot be built.")

    comparison = write_comparison_artifacts(
        rows=completed_rows,
        study_dir=study_dir,
    )
    if config.wandb_enabled:
        comparison_run_id = log_comparison_to_wandb(
            rows=completed_rows,
            plot_paths=comparison["artifacts"]["plots"],
            project=config.wandb_project,
            entity=config.wandb_entity,
            group=group,
            mode=config.wandb_mode,
            run_name=f"{study_id}-comparison",
            config=asdict(config),
        )
        comparison["wandb_comparison_run_id"] = comparison_run_id
        _atomic_json(study_dir / "comparison_summary.json", comparison)

    manifest["comparison"] = comparison
    manifest["completed_at"] = datetime.now().isoformat(timespec="seconds")
    _atomic_json(study_dir / "study_manifest.json", manifest)

    print("\n=== Experiment matrix completed ===")
    print(f"Study: {study_id}")
    print(f"W&B group: {group}")
    print(f"Completed experiments: {len(completed_rows)}")
    print(f"Best validation mAP experiment: {comparison['best_experiment_id']}")
    print(f"Outputs: {study_dir}")
    print(f"Comparison CSV: {comparison['artifacts']['csv']}")


if __name__ == "__main__":
    main()
