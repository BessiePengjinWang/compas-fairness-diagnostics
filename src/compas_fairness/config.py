from pathlib import Path

RANDOM_STATE = 42
TARGET = "two_year_recid"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = REPOSITORY_ROOT / "data" / "compas-scores-two-years.csv"
