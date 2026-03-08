"""Stage 1 — Load and split raw data."""
from __future__ import annotations

import pandas as pd

from src.data.loader import load_raw
from src.data.preprocessor import clean, encode_target
from src.pipeline.types import DataBundle


def load_full_df() -> pd.DataFrame:
    """Load, clean, encode target — no split.

    Shared by stage_load (single split) and cross_validate (k-fold) so any
    change to loading logic applies to both paths automatically.
    """
    df_raw, _ = load_raw()
    return encode_target(clean(df_raw))


def stage_load(config: dict) -> DataBundle:
    """Load raw data, clean, encode target, and split train/val(/test).

    Config keys:
        split_ratio (float, default 0.8): fraction of non-test data used for training.
        test_ratio  (float, default 0.0): fraction of total data held out as test set.
                                          0.0 means no test set (backward-compatible).

    Split order (sequential, no shuffle):
        1. Last test_ratio fraction → df_test
        2. Remaining: first split_ratio fraction → df_train, rest → df_val
    """
    split_ratio = config.get("split_ratio", 0.8)
    test_ratio  = config.get("test_ratio", 0.0)

    df = load_full_df()

    if test_ratio > 0:
        test_cut       = int((1.0 - test_ratio) * len(df))
        df_test        = df.iloc[test_cut:].copy()
        df             = df.iloc[:test_cut]
    else:
        df_test = None

    split = int(split_ratio * len(df))
    return DataBundle(
        df_train=df.iloc[:split].copy(),
        df_val=df.iloc[split:].copy(),
        df_test=df_test,
    )
