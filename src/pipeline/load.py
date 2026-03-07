"""Stage 1 — Load and split raw data."""
from __future__ import annotations

from src.data.loader import load_raw
from src.data.preprocessor import clean, encode_target
from src.pipeline.types import DataBundle


def stage_load(config: dict) -> DataBundle:
    """Load raw data, clean, encode target, and split train/val.

    Config keys:
        split_ratio (float, default 0.8): fraction used for training.
    """
    split_ratio = config.get("split_ratio", 0.8)

    df_raw, _ = load_raw()
    df = encode_target(clean(df_raw))

    split = int(split_ratio * len(df))
    return DataBundle(
        df_train=df.iloc[:split].copy(),
        df_val=df.iloc[split:].copy(),
    )
