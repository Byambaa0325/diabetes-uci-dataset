"""Improved feature set — identical to features.py with improvements from manual analysis:

1. admission_source_id: grouped into clinical categories and one-hot encoded
   rather than treated as a numeric ordinal. The raw IDs (1-25) are categorical
   codes with no ordinal relationship 

   Groups:
       referral  — physician (1), clinic (2), HMO (3)
       transfer  — hospital (4), SNF (5), other facility (6),
                   critical access (10), home health (18),
                   same hospital (22), ambulatory surgery (25)
       emergency — emergency room (7)
       other     — not available / null / unknown (9, 15, 17, 20, 21, etc.)

2. race: rare categories (Asian 1%, Hispanic 3%, Other 2%) are collapsed
   into a single "minority_other" group before one-hot encoding. 

   Groups: Caucasian (~75%), AfricanAmerican (~19%), minority_other (~6%)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Import everything shared with features.py
from src.data.features import (
    _AGE_ORDER, _A1C_ORDER, _GLU_ORDER, _MED_DIRECTION, _MED_COLS,
    _DISCHARGE_MAP, _DISCHARGE_DEFAULT,
    _ADMISSION_MAP, _ADMISSION_DEFAULT,
    _ICD_GROUPS, _TOP_SPECIALTIES, _NUMERIC_COLS,
    _encode_diagnoses, _encode_lab_results,
    _encode_medications, _encode_specialty, _add_engineered_features,
)

# Majority races kept separate; rare categories merged into one group
# so importance estimates are based on adequate sample sizes.
_RACE_MAJORITY = {"Caucasian", "AfricanAmerican"}


def _encode_demographics_improved(df: pd.DataFrame) -> pd.DataFrame:
    df["age"] = df["age"].map(_AGE_ORDER).fillna(4).astype(np.int8)
    df["gender"] = (df["gender"] == "Female").astype(np.int8)
    if "race" in df.columns:
        df["race"] = df["race"].apply(
            lambda x: x if (pd.notna(x) and x in _RACE_MAJORITY) else "minority_other"
        )
        df = pd.get_dummies(df, columns=["race"], drop_first=True, dtype=np.int8)
    return df

# ---------------------------------------------------------------------------
# Improved admission source grouping
# ---------------------------------------------------------------------------

_ADMISSION_SOURCE_MAP = {
    1: "referral",  2: "referral",  3: "referral",
    4: "transfer",  5: "transfer",  6: "transfer",
    10: "transfer", 18: "transfer", 22: "transfer", 25: "transfer",
    7: "emergency",
}
_ADMISSION_SOURCE_DEFAULT = "other"


def _encode_clinical_context_improved(df: pd.DataFrame) -> pd.DataFrame:
    """Clinical context encoding with corrected admission source grouping."""

    # Discharge disposition → grouped → one-hot (unchanged)
    if "discharge_disposition_id" in df.columns:
        df["discharge_group"] = (
            pd.to_numeric(df["discharge_disposition_id"], errors="coerce")
            .map(_DISCHARGE_MAP)
            .fillna(_DISCHARGE_DEFAULT)
        )
        df = pd.get_dummies(df, columns=["discharge_group"], drop_first=True, dtype=np.int8)
        df.drop(columns=["discharge_disposition_id"], inplace=True)

    # Admission type → grouped → one-hot (unchanged)
    if "admission_type_id" in df.columns:
        df["admission_group"] = (
            pd.to_numeric(df["admission_type_id"], errors="coerce")
            .map(_ADMISSION_MAP)
            .fillna(_ADMISSION_DEFAULT)
        )
        df = pd.get_dummies(df, columns=["admission_group"], drop_first=True, dtype=np.int8)
        df.drop(columns=["admission_type_id"], inplace=True)

    # Admission source → grouped → one-hot  (CHANGED from numeric passthrough)
    if "admission_source_id" in df.columns:
        df["admission_source_group"] = (
            pd.to_numeric(df["admission_source_id"], errors="coerce")
            .map(_ADMISSION_SOURCE_MAP)
            .fillna(_ADMISSION_SOURCE_DEFAULT)
        )
        df = pd.get_dummies(df, columns=["admission_source_group"], drop_first=True, dtype=np.int8)
        df.drop(columns=["admission_source_id"], inplace=True)

    # change, diabetesMed (unchanged)
    for col in ["change", "diabetesmed"]:
        if col in df.columns:
            df[col] = (df[col].str.lower() == "yes").astype(np.int8)

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_features_improved(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = True,
    columns: list | None = None,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, list]:
    """Encode and scale features with improved admission source encoding.

    Drop-in replacement for build_features — same signature, same return type.
    """
    df = df.copy()

    y = df.pop("readmitted").to_numpy(dtype=np.int32)

    df = _encode_demographics_improved(df)         # ← groups rare race categories
    df = _encode_clinical_context_improved(df)     # ← groups admission source
    df = _encode_diagnoses(df)
    df = _encode_lab_results(df)
    df = _encode_medications(df)
    df = _encode_specialty(df)
    df = _add_engineered_features(df)

    if columns is not None:
        df = df.reindex(columns=columns, fill_value=0)

    num_cols = [c for c in _NUMERIC_COLS if c in df.columns]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    if scaler is None:
        scaler = StandardScaler()
    if fit_scaler:
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        df[num_cols] = scaler.transform(df[num_cols])

    X = df.to_numpy(dtype=np.float32)
    return X, y, scaler, list(df.columns)
