import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "dataset" / "dataset_diabetes"


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (diabetic_data, ids_mapping) as raw DataFrames."""
    data = pd.read_csv(DATA_DIR / "diabetic_data.csv")
    mapping = pd.read_csv(DATA_DIR / "IDs_mapping.csv")
    return data, mapping
