from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def prepare_artifact_paths(
    output_dir: str | Path,
    filenames: Iterable[str],
    *,
    force: bool,
    artifact_name: str,
) -> tuple[Path, ...]:
    """Create an output directory and protect existing artifacts."""

    output_path = Path(output_dir)
    paths = tuple(output_path / filename for filename in filenames)
    existing = [path for path in paths if path.exists()]

    if existing and not force:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"{artifact_name} artifacts already exist. "
            f"Use --force to replace them: {formatted}"
        )

    output_path.mkdir(parents=True, exist_ok=True)
    return paths


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json_atomic(data: Any, path: str | Path) -> None:
    destination = Path(path)
    temporary = destination.with_name(f"{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    temporary.replace(destination)


def save_numpy_atomic(array: np.ndarray, path: str | Path) -> None:
    destination = Path(path)
    temporary = destination.with_name(f"{destination.name}.tmp")
    with temporary.open("wb") as file:
        np.save(file, array)
    temporary.replace(destination)


def save_csv_atomic(
    rows: list[dict[str, Any]],
    fieldnames: list[str],
    path: str | Path,
) -> None:
    destination = Path(path)
    temporary = destination.with_name(f"{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)
