"""Central paths shared by all Exercise 3 modules."""

from pathlib import Path


EXERCISE_DIR = Path(__file__).resolve().parent
DLA_LAB1_DIR = EXERCISE_DIR.parent

# The YAML files use paths such as ``Exercise3/...`` and ``Exercise2/...``.
# Those paths are relative to the DLA_LAB1 directory, which is also the
# directory from which commands such as ``python -m Exercise3...`` are run.
PROJECT_ROOT = DLA_LAB1_DIR

# Keep a separate name for the parent repository directory, when needed.
REPOSITORY_ROOT = DLA_LAB1_DIR.parent

DATA_DIR = DLA_LAB1_DIR / "data"
OUTPUT_DIR = EXERCISE_DIR / "outputs"
