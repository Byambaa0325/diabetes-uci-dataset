"""Stage 2 — Build feature arrays from split DataFrames."""
from __future__ import annotations

import numpy as np
from sklearn.utils import resample

from src.data.features import build_features
from src.data.features_basic import build_features_basic
from src.data.features_improved import build_features_improved
from src.pipeline.types import DataBundle, FeatureBundle

_FEATURIZERS = {
    "full":     build_features,           # original — admission_source_id as numeric
    "basic":    build_features_basic,     # minimal encoding (~18 features)
    "improved": build_features_improved,  # grouped admission_source_id + collapsed race
}


def stage_featurize(data: DataBundle, config: dict) -> FeatureBundle:
    """Encode and scale features; optionally subsample or select features.

    Config keys:
        featurizer        (str,  default "full"):  which feature set to use.
        subsample         (int,  default None):    cap on training rows (stratified).
        selected_features (list, default None):    subset of feature names to keep.
                                                   Applied after encoding — use with
                                                   permutation importance rankings.
    """
    featurizer_key = config.get("featurizer", "full")
    if featurizer_key not in _FEATURIZERS:
        raise ValueError(f"Unknown featurizer '{featurizer_key}'. Choose from: {list(_FEATURIZERS)}")
    build_fn = _FEATURIZERS[featurizer_key]

    # Fit scaler on train; column-align val/test to prevent fold mismatch
    X_train, y_train, scaler, train_cols = build_fn(data.df_train, fit_scaler=True)
    X_val,   y_val,   _,      _          = build_fn(data.df_val,   scaler=scaler, fit_scaler=False, columns=train_cols)

    if data.df_test is not None:
        X_test, y_test, _, _ = build_fn(data.df_test, scaler=scaler, fit_scaler=False, columns=train_cols)
    else:
        X_test, y_test = None, None

    # Optional: keep only a selected subset of features
    selected = config.get("selected_features")
    if selected:
        idx = [i for i, c in enumerate(train_cols) if c in set(selected)]
        if not idx:
            raise ValueError("selected_features produced an empty feature set — check column names")
        X_train    = X_train[:, idx]
        X_val      = X_val[:, idx]
        X_test     = X_test[:, idx] if X_test is not None else None
        train_cols = [train_cols[i] for i in idx]

    subsample = config.get("subsample")
    if subsample and subsample < len(X_train):
        X_train, y_train = resample(
            X_train, y_train,
            n_samples=subsample,
            stratify=y_train,
            random_state=42,
        )

    return FeatureBundle(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_features=X_val.shape[1],
        scaler=scaler,
        X_test=X_test,
        y_test=y_test,
        feature_names=train_cols,
    )
