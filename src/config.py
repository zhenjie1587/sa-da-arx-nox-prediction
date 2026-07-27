"""Central experiment configuration.

All random choices and data-split boundaries are fixed here so that every
table and figure can be reproduced from a single command.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = (
    PROJECT_ROOT
    / "source_data"
    / "Data for Dynamic modeling of NOX emission in a 660 MW coal-fired boiler with LSTM"
    / "data.xlsx"
)
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RANDOM_SEED = 42
MAX_EXOGENOUS_LAG = 10
TARGET_LAGS = tuple(range(1, 11))

# Zero-based row positions in the 10,000-sample original data sheet.
# Samples 1--10 are not used because delayed features require warm-up.
# The final test period starts at sample 7002 to match the source LSTM
# predictions stored in the workbook.
TRAIN_START = MAX_EXOGENOUS_LAG
TRAIN_END = 6500
CALIBRATION_START = 6500
CALIBRATION_END = 7001
TEST_START = 7001
TEST_END = 10000

RIDGE_ALPHA = 1.0
TRAIN_DROPOUT_RATES = (0.10, 0.20, 0.30)
SCENARIO_REPLICATES = 5
CONFORMAL_ALPHA = 0.05

# The source workbook contains no column names. The first numeric column is a
# sequence number, followed by 27 process measurements and two NOx channels.
PROCESS_COLUMNS = tuple(f"F{i:02d}" for i in range(1, 28))
TARGET_COLUMNS = ("NOx_A", "NOx_B")
LOAD_PROXY_COLUMN = "F01"
