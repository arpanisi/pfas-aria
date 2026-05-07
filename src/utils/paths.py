"""
Centralized path resolution.
Import PROJECT_ROOT or any sub-path from here.
Never hardcode paths in any other module.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CORPUS_DIR = DATA_DIR / "corpus"
REPORTS_DIR = DATA_DIR / "outputs" / "reports"
OUTPUTS_DIR = DATA_DIR / "outputs"

CONFIGS_DIR = PROJECT_ROOT / "configs"
LOGS_DIR = OUTPUTS_DIR / "logs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
CHROMA_DIR = OUTPUTS_DIR / "chroma"
MLFLOW_DIR = PROJECT_ROOT / "mlflow"


def ensure_dirs() -> None:
    """Create all output directories if they don't exist.
    Call once at pipeline startup."""
    for d in [
        RAW_DIR,
        PROCESSED_DIR,
        CORPUS_DIR,
        OUTPUTS_DIR,
        LOGS_DIR,
        REPORTS_DIR,
        CHROMA_DIR,
        MLFLOW_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
