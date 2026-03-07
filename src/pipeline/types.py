"""Typed contracts passed between pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class DataBundle:
    """Raw cleaned DataFrames after loading and splitting."""
    df_train: pd.DataFrame
    df_val:   pd.DataFrame


@dataclass
class FeatureBundle:
    """Encoded, scaled feature arrays ready for training."""
    X_train:    np.ndarray
    y_train:    np.ndarray
    X_val:      np.ndarray
    y_val:      np.ndarray
    n_features: int
    scaler:     object   # fitted StandardScaler


@dataclass
class PipelineResult:
    """Collected outputs from a completed run."""
    run_dir:        Path
    model:          object
    metrics:        dict
    feature_bundle: FeatureBundle
    elapsed_s:      float
