from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

CLSI_PATH = RAW_DIR / "CLSI.csv"

INPUT_CODE_COL = "Species Code"
INPUT_NAME_COL = "Species Full Name"

TARGET_COLS = [
    "Reported Name",
    "CLSI Breakpoint Group",
    "Bacteria Gram Stain",
]

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
NOISY_COPIES_PER_ROW = 2
STRATIFY_COLS = ["CLSI Breakpoint Group"]
STRATIFY_MIN_COUNT = 2
