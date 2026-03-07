"""Stage 2 — Build feature arrays from split DataFrames."""
from __future__ import annotations

from sklearn.utils import resample

from src.data.features import build_features
from src.data.features_basic import build_features_basic
from src.pipeline.types import DataBundle, FeatureBundle

_FEATURIZERS = {
    "full":  build_features,
    "basic": build_features_basic,
}


def stage_featurize(data: DataBundle, config: dict) -> FeatureBundle:
    """Encode and scale features; optionally subsample the training set.

    Config keys:
        featurizer (str, default "full"): "full" for full feature engineering,
                                          "basic" for minimal as-is encoding.
        subsample  (int, default None):   stratified subsample size for X_train.
                                          Val set is always the full holdout.
    """
    featurizer_key = config.get("featurizer", "full")
    if featurizer_key not in _FEATURIZERS:
        raise ValueError(f"Unknown featurizer '{featurizer_key}'. Choose from: {list(_FEATURIZERS)}")
    build_fn = _FEATURIZERS[featurizer_key]

    X_train, y_train, scaler = build_fn(data.df_train, fit_scaler=True)
    X_val,   y_val,   _      = build_fn(data.df_val, scaler=scaler, fit_scaler=False)

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
        n_features=X_val.shape[1],  # full val reflects true feature count
        scaler=scaler,
    )
