"""Run the ordered Exercise 3 validation suite."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from Exercise3.models.gtsrb_transfer import resolve_project_path
from Exercise3.paths import OUTPUT_DIR


DEFAULT_CHECKPOINT = Path(
    "Exercise3/checkpoints/gtsrb_resnet50_full_linear.pt"
)
DEFAULT_SUMMARY = OUTPUT_DIR / "checks" / "run_all_summary.json"


@dataclass(frozen=True)
class CheckResult:
    name: str
    module: str
    command: list[str]
    status: str
    return_code: int | None
    seconds: float
    reason: str | None = None


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run dataset, transform, DataLoader, visualization, model, "
            "smoke-test and optional GTSRB-transfer checks in order."
        )
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device forwarded to model and smoke checks.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers forwarded to relevant checks.",
    )
    parser.add_argument(
        "--weights",
        choices=("coco", "none"),
        default="coco",
        help=(
            "Detector weights for structural checks. Use none only for "
            "offline structural validation."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Canonical GTSRB checkpoint used by the transfer check.",
    )
    parser.add_argument(
        "--require-gtsrb",
        action="store_true",
        help="Fail if the GTSRB checkpoint is missing.",
    )
    parser.add_argument(
        "--skip-gtsrb",
        action="store_true",
        help="Do not run the GTSRB transfer check.",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip the real forward/backward smoke test.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Run remaining checks after a failure.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="JSON summary path.",
    )
    return parser.parse_args()


def _command(module: str, *arguments: str) -> list[str]:
    return [sys.executable, "-m", module, *arguments]


def _check_commands(
    *,
    device: str,
    num_workers: int,
    weights: str,
    checkpoint: Path,
    skip_smoke: bool,
    skip_gtsrb: bool,
) -> list[tuple[str, str, list[str], str | None]]:
    checks: list[tuple[str, str, list[str], str | None]] = [
        (
            "adapter",
            "Exercise3.checks.validate_adapter",
            _command("Exercise3.checks.validate_adapter"),
            None,
        ),
        (
            "transforms",
            "Exercise3.checks.validate_transforms",
            _command("Exercise3.checks.validate_transforms"),
            None,
        ),
        (
            "dataloaders",
            "Exercise3.checks.validate_loaders",
            _command(
                "Exercise3.checks.validate_loaders",
                "--num-workers",
                str(num_workers),
            ),
            None,
        ),
        (
            "ground-truth-visualization",
            "Exercise3.checks.validate_ground_truth",
            _command("Exercise3.checks.validate_ground_truth"),
            None,
        ),
        (
            "model",
            "Exercise3.checks.validate_model",
            _command(
                "Exercise3.checks.validate_model",
                "--device",
                device,
                "--weights",
                weights,
                "--no-progress",
            ),
            None,
        ),
    ]

    if skip_smoke:
        checks.append(
            (
                "detector-smoke",
                "Exercise3.checks.smoke_test_detector",
                [],
                "skipped by --skip-smoke",
            )
        )
    else:
        checks.append(
            (
                "detector-smoke",
                "Exercise3.checks.smoke_test_detector",
                _command(
                    "Exercise3.checks.smoke_test_detector",
                    "--device",
                    device,
                    "--weights",
                    weights,
                    "--num-workers",
                    str(num_workers),
                    "--no-progress",
                ),
                None,
            )
        )

    if skip_gtsrb:
        checks.append(
            (
                "gtsrb-transfer",
                "Exercise3.checks.validate_gtsrb_transfer",
                [],
                "skipped by --skip-gtsrb",
            )
        )
    elif checkpoint.is_file():
        checks.append(
            (
                "gtsrb-transfer",
                "Exercise3.checks.validate_gtsrb_transfer",
                _command(
                    "Exercise3.checks.validate_gtsrb_transfer",
                    "--checkpoint",
                    str(checkpoint),
                    "--required-strategy",
                    "full",
                    "--no-progress",
                ),
                None,
            )
        )
    else:
        checks.append(
            (
                "gtsrb-transfer",
                "Exercise3.checks.validate_gtsrb_transfer",
                [],
                f"checkpoint not found: {checkpoint}",
            )
        )

    return checks


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    arguments = parse_arguments()
    if arguments.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if arguments.skip_gtsrb and arguments.require_gtsrb:
        raise ValueError(
            "--skip-gtsrb and --require-gtsrb cannot be used together."
        )

    checkpoint = resolve_project_path(arguments.checkpoint).resolve()
    summary_path = arguments.summary.expanduser()
    if not summary_path.is_absolute():
        summary_path = resolve_project_path(summary_path).resolve()

    checks = _check_commands(
        device=arguments.device,
        num_workers=arguments.num_workers,
        weights=arguments.weights,
        checkpoint=checkpoint,
        skip_smoke=arguments.skip_smoke,
        skip_gtsrb=arguments.skip_gtsrb,
    )
    results: list[CheckResult] = []
    failed = False

    print("\n=== Exercise 3 ordered validation suite ===")
    for name, module, command, skip_reason in checks:
        if skip_reason is not None:
            if (
                name == "gtsrb-transfer"
                and arguments.require_gtsrb
                and not arguments.skip_gtsrb
            ):
                results.append(
                    CheckResult(
                        name=name,
                        module=module,
                        command=command,
                        status="failed",
                        return_code=None,
                        seconds=0.0,
                        reason=skip_reason,
                    )
                )
                failed = True
                print(f"\n[{name}] FAILED: {skip_reason}")
                if not arguments.continue_on_error:
                    break
                continue

            results.append(
                CheckResult(
                    name=name,
                    module=module,
                    command=command,
                    status="skipped",
                    return_code=None,
                    seconds=0.0,
                    reason=skip_reason,
                )
            )
            print(f"\n[{name}] SKIPPED: {skip_reason}")
            continue

        print(f"\n[{name}]")
        print("$ " + subprocess.list2cmdline(command), flush=True)
        if arguments.dry_run:
            results.append(
                CheckResult(
                    name=name,
                    module=module,
                    command=command,
                    status="dry-run",
                    return_code=None,
                    seconds=0.0,
                )
            )
            continue

        started = perf_counter()
        completed = subprocess.run(command, check=False)
        seconds = perf_counter() - started
        status = "passed" if completed.returncode == 0 else "failed"
        results.append(
            CheckResult(
                name=name,
                module=module,
                command=command,
                status=status,
                return_code=int(completed.returncode),
                seconds=float(seconds),
            )
        )
        print(f"[{name}] {status.upper()} in {seconds:.2f}s")

        if completed.returncode != 0:
            failed = True
            if not arguments.continue_on_error:
                break

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": arguments.device,
        "num_workers": arguments.num_workers,
        "weights": arguments.weights,
        "checkpoint": str(checkpoint),
        "require_gtsrb": arguments.require_gtsrb,
        "dry_run": arguments.dry_run,
        "results": [asdict(result) for result in results],
        "all_required_checks_passed": not failed,
    }
    _write_summary(summary_path, payload)

    print("\n=== Validation summary ===")
    for result in results:
        print(f"{result.name}: {result.status}")
    print(f"Summary: {summary_path}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
