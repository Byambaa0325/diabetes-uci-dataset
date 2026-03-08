"""Stage 2 — Build feature arrays from split DataFrames."""
from __future__ import annotations

from sklearn.utils import resample

from src.data.features import build_features
from src.data.features_basic import build_features_basic
from src.data.features_improved import build_features_improved
from src.pipeline.types import DataBundle, FeatureBundle

_FEATURIZERS = {
    "full":     build_features,           # original — admission_source_id as numeric
    "basic":    build_features_basic,     # minimal encoding (~18 features)
    "improved": build_features_improved,  # grouped admission_source_id one-hot
}


def stage_featurize(data: DataBundle, config: dict) -> FeatureBundle:
    """Encode and scale features; optionally subsample the training set.

    Config keys:
        featurizer (str, default "full"): "full" for full feature engineering,
                                          "basic" for minimal as-is encoding.
        subsample  (int, default None):   stratified subsample size for X_train.
                                          Val and test sets are never subsampled.
    """
    featurizer_key = config.get("featurizer", "full")
    if featurizer_key not in _FEATURIZERS:
        raise ValueError(f"Unknown featurizer '{featurizer_key}'. Choose from: {list(_FEATURIZERS)}")
    build_fn = _FEATURIZERS[featurizer_key]

    # Fit scaler and capture training column schema in one call.
    # train_cols is used to align val/test — prevents feature count mismatches
    # when rare get_dummies categories only appear in some CV folds.
    X_train, y_train, scaler, train_cols = build_fn(data.df_train, fit_scaler=True)
    X_val,   y_val,   _,      _          = build_fn(data.df_val,   scaler=scaler, fit_scaler=False, columns=train_cols)

    # Transform test set with the same scaler and column schema — never fit on it
    if data.df_test is not None:
        X_test, y_test, _, _ = build_fn(data.df_test, scaler=scaler, fit_scaler=False, columns=train_cols)
    else:
        X_test, y_test = None, None

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
    )
