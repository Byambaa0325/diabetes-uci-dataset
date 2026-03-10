import numpy as np
import pandas as pd

# discharge_disposition_id codes for death/hospice — removed per Strack et al. (2014)
# to avoid biasing the readmission label (these patients are always readmitted=NO by construction)
_EXPIRED_DISCHARGE_CODES = {11, 13, 14, 19, 20, 21}

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
    if "discharge_disposition_id" in df.columns:
        df = df[~pd.to_numeric(df["discharge_disposition_id"], errors="coerce")
                  .isin(_EXPIRED_DISCHARGE_CODES)]
    if "encounter_id" in df.columns:
        df = df.sort_values("encounter_id")
    df.drop_duplicates(subset="patient_nbr", keep="first", inplace=True)
    df.drop(columns=[c for c in _DROP_COLS if c in df.columns], inplace=True)
    return df


def encode_target(df: pd.DataFrame, col: str = "readmitted", mode: str = "any") -> pd.DataFrame:
    """Binary encode readmission.

    mode='any'  — any readmission (<30 or >30) → 1, NO → 0  [Tasks 2-4, default]
    mode='lt30' — early readmission only (<30) → 1, else → 0  [Task 5]
    """
    df = df.copy()
    if mode == "lt30":
        df[col] = (df[col] == "<30").astype(int)
    else:
        df[col] = (df[col] != "NO").astype(int)
    return df
