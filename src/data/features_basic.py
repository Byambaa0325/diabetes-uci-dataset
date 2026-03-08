from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

_AGE_ORDER = {
    "[0-10)": 0, "[10-20)": 1, "[20-30)": 2, "[30-40)": 3, "[40-50)": 4,
    "[50-60)": 5, "[60-70)": 6, "[70-80)": 7, "[80-90)": 8, "[90-100)": 9,
}
_A1C_ORDER = {np.nan: 0, "None": 0, "Norm": 1, ">7": 2, ">8": 3}
_GLU_ORDER = {np.nan: 0, "None": 0, "Norm": 1, ">200": 2, ">300": 3}

_NUMERIC_COLS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]

_DROP_COLS = [
    "diag_1", "diag_2", "diag_3",
    "medical_specialty",
    "discharge_disposition_id",
    "admission_type_id",
    "admission_source_id",
    # all medication columns
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "tolazamide", "insulin",
    "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]


def build_features_basic(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = True,
    columns: list | None = None,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, list]:
    """Minimal encode and scale. Returns (X, y, scaler, columns).

    Pass a fitted scaler with fit_scaler=False to transform val/test sets
    using the same scale as the training set. Pass columns from the training
    call to align val/test and prevent feature count mismatches.
    """
    df = df.copy()

    y = df.pop("readmitted").to_numpy(dtype=np.int32)

    # Drop complex/high-cardinality columns
    df.drop(columns=[c for c in _DROP_COLS if c in df.columns], inplace=True)

    # Ordinal
    if "age" in df.columns:
        df["age"] = df["age"].map(_AGE_ORDER).fillna(4).astype(np.int8)
    if "a1cresult" in df.columns:
        df["a1cresult"] = df["a1cresult"].map(_A1C_ORDER).fillna(0).astype(np.int8)
    if "max_glu_serum" in df.columns:
        df["max_glu_serum"] = df["max_glu_serum"].map(_GLU_ORDER).fillna(0).astype(np.int8)

    # Binary
    if "gender" in df.columns:
        df["gender"] = (df["gender"] == "Female").astype(np.int8)
    for col in ["change", "diabetesmed"]:
        if col in df.columns:
            df[col] = (df[col].str.lower() == "yes").astype(np.int8)

    # One-hot: race only
    if "race" in df.columns:
        df = pd.get_dummies(df, columns=["race"], drop_first=True, dtype=np.int8)

    # Align columns to training set
    if columns is not None:
        df = df.reindex(columns=columns, fill_value=0)

    # Scale numeric columns
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
