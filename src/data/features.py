from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

_AGE_ORDER = {
    "[0-10)": 0, "[10-20)": 1, "[20-30)": 2, "[30-40)": 3, "[40-50)": 4,
    "[50-60)": 5, "[60-70)": 6, "[70-80)": 7, "[80-90)": 8, "[90-100)": 9,
}

# A1Cresult: None (not measured)=0, Norm=1, >7=2, >8=3
_A1C_ORDER = {np.nan: 0, "None": 0, "Norm": 1, ">7": 2, ">8": 3}

# max_glu_serum: None=0, Norm=1, >200=2, >300=3
_GLU_ORDER = {np.nan: 0, "None": 0, "Norm": 1, ">200": 2, ">300": 3}

# Medication change direction: No=0, Steady=1, Down=2, Up=3
_MED_DIRECTION = {"No": 0, "Steady": 1, "Down": 2, "Up": 3}

# Medication columns with directional encoding
_MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "tolazamide", "insulin",
    "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

# Discharge disposition: clinical groupings
# 1=home, 2-5=transfer/facility, 6=home+health, 7=AMA, 11/19/20/21=expired/hospice
_DISCHARGE_MAP = {
    1: "home", 6: "home", 8: "home",
    2: "transfer", 3: "transfer", 4: "transfer", 5: "transfer",
    9: "transfer", 10: "transfer", 12: "transfer", 15: "transfer",
    16: "transfer", 17: "transfer", 22: "transfer", 23: "transfer", 24: "transfer",
    7: "ama",
    11: "expired", 19: "expired", 20: "expired", 21: "expired",
}
_DISCHARGE_DEFAULT = "other"

# Admission type: clinical groupings
_ADMISSION_MAP = {1: "emergency", 2: "urgent", 3: "elective"}
_ADMISSION_DEFAULT = "other"

# ICD-9 code → primary condition group (Strack et al. 2014 CCS-inspired)
_ICD_GROUPS = [
    ("Circulatory",      lambda c: (390 <= c < 460) or c == 785),
    ("Respiratory",      lambda c: (460 <= c < 520) or c == 786),
    ("Digestive",        lambda c: (520 <= c < 580) or c == 787),
    ("Diabetes",         lambda c: 250 <= c < 251),
    ("Injury",           lambda c: 800 <= c < 1000),
    ("Musculoskeletal",  lambda c: 710 <= c < 740),
    ("Genitourinary",    lambda c: (580 <= c < 630) or c == 788),
    ("Neoplasms",        lambda c: 140 <= c < 240),
]

# Medical specialty: keep top specialties, group rest as Other
_TOP_SPECIALTIES = {
    "InternalMedicine", "Cardiology", "Surgery-General", "Orthopedics",
    "Gastroenterology", "Nephrology", "Pulmonology", "Hematology/Oncology",
    "ObstetricsandGynecology", "Urology", "Psychiatry", "Family/GeneralPractice",
}

# Numeric columns (scaled)
_NUMERIC_COLS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = True,
    columns: list | None = None,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, list]:
    """Encode and scale features. Returns (X, y, scaler, columns).

    Pass a fitted scaler with fit_scaler=False to transform val/test sets
    using the same scale as the training set. Pass the columns list returned
    from the training call to align val/test columns and prevent feature
    count mismatches from rare get_dummies categories.
    """
    df = df.copy()

    y = df.pop("readmitted").to_numpy(dtype=np.int32)

    df = _encode_demographics(df)
    df = _encode_clinical_context(df)
    df = _encode_diagnoses(df)
    df = _encode_lab_results(df)
    df = _encode_medications(df)
    df = _encode_specialty(df)
    df = _add_engineered_features(df)

    # Align columns to training set � prevents mismatch when rare categories
    # only appear in one split (e.g. during cross-validation folds).
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


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def _encode_demographics(df: pd.DataFrame) -> pd.DataFrame:
    df["age"] = df["age"].map(_AGE_ORDER).fillna(4).astype(np.int8)
    df["gender"] = (df["gender"] == "Female").astype(np.int8)
    df = pd.get_dummies(df, columns=["race"], drop_first=True, dtype=np.int8)
    return df


def _encode_clinical_context(df: pd.DataFrame) -> pd.DataFrame:
    # Discharge disposition → grouped categories → one-hot
    if "discharge_disposition_id" in df.columns:
        df["discharge_group"] = (
            pd.to_numeric(df["discharge_disposition_id"], errors="coerce")
            .map(_DISCHARGE_MAP)
            .fillna(_DISCHARGE_DEFAULT)
        )
        df = pd.get_dummies(df, columns=["discharge_group"], drop_first=True, dtype=np.int8)
        df.drop(columns=["discharge_disposition_id"], inplace=True)

    # Admission type → grouped → one-hot
    if "admission_type_id" in df.columns:
        df["admission_group"] = (
            pd.to_numeric(df["admission_type_id"], errors="coerce")
            .map(_ADMISSION_MAP)
            .fillna(_ADMISSION_DEFAULT)
        )
        df = pd.get_dummies(df, columns=["admission_group"], drop_first=True, dtype=np.int8)
        df.drop(columns=["admission_type_id"], inplace=True)

    # Admission source: keep as numeric (1-25)
    # Note: these are categorical codes, not ordinal — see features_improved.py
    # for a corrected grouped encoding.
    if "admission_source_id" in df.columns:
        df["admission_source_id"] = pd.to_numeric(
            df["admission_source_id"], errors="coerce"
        ).fillna(df["admission_source_id"].median())

    # change, diabetesMed
    for col in ["change", "diabetesmed"]:
        if col in df.columns:
            df[col] = (df[col].str.lower() == "yes").astype(np.int8)

    return df


def _encode_diagnoses(df: pd.DataFrame) -> pd.DataFrame:
    """Map ICD-9 codes in diag_1/2/3 to clinical groups, then one-hot."""
    for diag_col in ["diag_1", "diag_2", "diag_3"]:
        if diag_col not in df.columns:
            continue
        group_col = f"{diag_col}_group"
        df[group_col] = df[diag_col].apply(_icd_to_group)
        df = pd.get_dummies(df, columns=[group_col], drop_first=True, dtype=np.int8)
        df.drop(columns=[diag_col], inplace=True)
    return df


def _icd_to_group(code) -> str:
    if pd.isna(code):
        return "Unknown"
    code = str(code).strip()
    if code.startswith("V") or code.startswith("E"):
        return "External"
    try:
        c = float(code)
        for name, test in _ICD_GROUPS:
            if test(c):
                return name
        return "Other"
    except ValueError:
        return "Other"


def _encode_lab_results(df: pd.DataFrame) -> pd.DataFrame:
    # A1Cresult: ordinal 0–3 (not measured → Norm → >7 → >8)
    if "a1cresult" in df.columns:
        df["a1cresult"] = df["a1cresult"].map(_A1C_ORDER).fillna(0).astype(np.int8)

    # max_glu_serum: ordinal 0–3
    if "max_glu_serum" in df.columns:
        df["max_glu_serum"] = df["max_glu_serum"].map(_GLU_ORDER).fillna(0).astype(np.int8)

    return df


def _encode_medications(df: pd.DataFrame) -> pd.DataFrame:
    """Encode medication columns with direction: No=0, Steady=1, Down=2, Up=3."""
    for col in [c for c in _MED_COLS if c in df.columns]:
        df[col] = df[col].map(_MED_DIRECTION).fillna(0).astype(np.int8)
    return df


def _encode_specialty(df: pd.DataFrame) -> pd.DataFrame:
    if "medical_specialty" not in df.columns:
        return df
    df["medical_specialty"] = df["medical_specialty"].apply(
        lambda x: x if (pd.notna(x) and x in _TOP_SPECIALTIES) else ("Missing" if pd.isna(x) else "Other")
    )
    df = pd.get_dummies(df, columns=["medical_specialty"], drop_first=True, dtype=np.int8)
    return df


def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Interaction and aggregate features motivated by EDA + domain knowledge."""

    # Total prior hospital contact (strongest single signal from EDA)
    num = lambda c: pd.to_numeric(df[c], errors="coerce").fillna(0) if c in df.columns else 0
    df["prior_hospital_load"] = num("number_inpatient") + num("number_emergency") + num("number_outpatient")

    # Total medications prescribed across all drug columns
    med_present = [c for c in _MED_COLS if c in df.columns]
    if med_present:
        df["total_meds_prescribed"] = (df[med_present] > 0).sum(axis=1).astype(np.int8)

    # Lab intensity relative to procedures (proxy for diagnostic complexity)
    df["lab_to_procedure_ratio"] = (
        num("num_lab_procedures") / (num("num_procedures") + 1)
    )

    # Age × medication burden (older patients on many drugs = higher risk)
    if "age" in df.columns:
        df["age_x_medications"] = (
            df["age"].astype(float) * num("num_medications")
        )

    # Was A1C measured and result was poor (>7 or >8) — clinical marker of uncontrolled diabetes
    if "a1cresult" in df.columns:
        df["a1c_poor_control"] = (df["a1cresult"] >= 2).astype(np.int8)

    # Insulin + change interaction — insulin adjustment suggests unstable glucose control
    if "insulin" in df.columns and "change" in df.columns:
        df["insulin_adjusted"] = (
            (df["insulin"] >= 2) & (df["change"] == 1)
        ).astype(np.int8)

    return df
