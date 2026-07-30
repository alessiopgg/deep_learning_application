"""Central paths shared by all Exercise 3 modules."""

from pathlib import Path


EXERCISE_DIR = Path(__file__).resolve().parent
DLA_LAB1_DIR = EXERCISE_DIR.parent
PROJECT_ROOT = DLA_LAB1_DIR.parent
DATA_DIR = DLA_LAB1_DIR / "data"
OUTPUT_DIR = EXERCISE_DIR / "outputs"
