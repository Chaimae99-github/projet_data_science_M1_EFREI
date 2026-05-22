from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "predictive_maintenance_v3.csv"


def load_raw_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the predictive maintenance dataset with a parsed timestamp column."""
    path = Path(path)
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df
