import numpy as np
import pandas as pd

# Columns dropped unconditionally before feature engineering
_DROP_COLS = [
    "encounter_id",       # row identifier, no signal
    "patient_nbr",        # patient identifier, no signal
    "weight",             # 97% missing
    "payer_code",         # 40% missing, weak prior relevance
    "examide", "citoglipton", "troglitazone",  # near-zero prescription rates
    # diag_1/2/3 and medical_specialty are kept — grouped in features.py
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names, replace missing markers, deduplicate on patient."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    df.replace("?", np.nan, inplace=True)
    df.drop_duplicates(subset="patient_nbr", keep="first", inplace=True)
    df.drop(columns=[c for c in _DROP_COLS if c in df.columns], inplace=True)
    return df


def encode_target(df: pd.DataFrame, col: str = "readmitted") -> pd.DataFrame:
    """Binary encode readmission: any readmission (<30 or >30) → 1, NO → 0."""
    df = df.copy()
    df[col] = (df[col] != "NO").astype(int)
    return df
