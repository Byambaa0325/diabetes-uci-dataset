import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, handle missing markers, normalise column names."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    df.replace("?", pd.NA, inplace=True)
    df.drop_duplicates(subset="patient_nbr", keep="first", inplace=True)
    return df


def encode_target(df: pd.DataFrame, col: str = "readmitted") -> pd.DataFrame:
    """Binary encode readmission: <30 → 1, else → 0."""
    df = df.copy()
    df[col] = (df[col] == "<30").astype(int)
    return df
