"""Typed contracts passed between pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class DataBundle:
    """Raw cleaned DataFrames after loading and splitting."""
    df_train: pd.DataFrame
    df_val:   pd.DataFrame
    df_test:  Optional[pd.DataFrame] = field(default=None)


@dataclass
class FeatureBundle:
    """Encoded, scaled feature arrays ready for training."""
    X_train:    np.ndarray
    y_train:    np.ndarray
    X_val:      np.ndarray
    y_val:      np.ndarray
    n_features: int
    scaler:     object                      # fitted StandardScaler
    X_test:     Optional[np.ndarray] = field(default=None)
    y_test:     Optional[np.ndarray] = field(default=None)


@dataclass
class PipelineResult:
    """Collected outputs from a completed run.

    Single-split:     fold_metrics=None, test_metrics=None (unless test_ratio set).
    Cross-validation: fold_metrics holds per-fold dicts; metrics holds means;
                      run_dir and model are None (no single artefact).
    """
    run_dir:        Optional[Path]
    model:          Optional[object]
    metrics:        dict            # val metrics (single) or CV means
    feature_bundle: Optional[FeatureBundle]
    elapsed_s:      float
    fold_metrics:   Optional[list[dict]] = field(default=None)
    test_metrics:   Optional[dict]       = field(default=None)

    @property
    def is_cv(self) -> bool:
        return self.fold_metrics is not None
