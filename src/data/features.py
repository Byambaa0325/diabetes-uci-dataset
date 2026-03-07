from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Ordinal mapping for age brackets
_AGE_ORDER = {
    "[0-10)": 0, "[10-20)": 1, "[20-30)": 2, "[30-40)": 3, "[40-50)": 4,
    "[50-60)": 5, "[60-70)": 6, "[70-80)": 7, "[80-90)": 8, "[90-100)": 9,
}

# Medication columns — encoded as: No → 0, Steady/Up/Down → 1
_MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "tolazamide", "insulin",
    "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

# Numeric columns kept as-is (imputed + scaled)
_NUMERIC_COLS = [
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]


def build_features(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = True,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Encode and scale features. Returns (X, y, scaler).

    Pass a fitted scaler with fit_scaler=False to transform val/test sets
    using the same scale as the training set.
    """
    df = df.copy()

    # --- Target ---
    y = df.pop("readmitted").to_numpy(dtype=np.int32)

    # --- Age: ordinal ---
    df["age"] = df["age"].map(_AGE_ORDER)

    # --- Gender: binary ---
    df["gender"] = (df["gender"] == "Female").astype(np.int8)

    # --- Race: one-hot (drop_first to avoid dummy trap) ---
    df = pd.get_dummies(df, columns=["race"], drop_first=True, dtype=np.int8)

    # --- A1Cresult, max_glu_serum: was-measured binary ---
    for col in ["a1cresult", "max_glu_serum"]:
        if col in df.columns:
            df[col] = df[col].notna().astype(np.int8)

    # --- Medications: prescribed or not ---
    for col in [c for c in _MED_COLS if c in df.columns]:
        df[col] = (df[col] != "No").astype(np.int8)

    # --- Binary flag columns ---
    for col in ["change", "diabetesmed"]:
        if col in df.columns:
            df[col] = (df[col].str.lower() == "yes").astype(np.int8)

    # --- Numeric: median imputation + standard scaling ---
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
    return X, y, scaler
