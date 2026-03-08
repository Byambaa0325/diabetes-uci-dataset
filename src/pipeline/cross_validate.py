"""K-fold cross-validation for the pipeline.

Featurization (scaler fit) is performed independently per fold so no
information from the validation fold contaminates the training fold.

Called internally by run_pipeline when config contains 'cv_splits'.
Also available directly for explicit CV workflows.

Public API
----------
cross_validate(config, n_splits, random_state)
    -> {'fold_metrics': [...], 'summary': {'roc_auc_mean': ..., ...}}

cross_validate_sweep(base_config, param_grid, n_splits, random_state)
    -> list of cross_validate results, one per param_grid entry.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.pipeline.load import load_full_df
from src.pipeline.types import DataBundle, FeatureBundle
from src.pipeline.featurize import stage_featurize
from src.pipeline.run_train import stage_train
from src.pipeline.run_evaluate import stage_evaluate

_SCALAR_KEYS = ("roc_auc", "f1", "recall", "recall_lt30", "precision", "threshold")


def cross_validate(
    config: dict,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """Stratified k-fold cross-validation over the full dataset.

    Args:
        config:       Pipeline config dict (same format as run_pipeline).
        n_splits:     Number of folds.
        random_state: Seed for fold splitting.

    Returns:
        Dict with:
            fold_metrics  — list of metric dicts, one per fold.
            summary       — {metric_mean, metric_std} for each scalar metric.
    """
    df = load_full_df()

    # Exclude the held-out test set from CV so folds never see it
    test_ratio = config.get("test_ratio", 0.0)
    if test_ratio > 0:
        test_cut = int((1.0 - test_ratio) * len(df))
        df = df.iloc[:test_cut].copy()

    y_all = df["readmitted"].values
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df, y_all)):
        print(f"  fold {fold + 1}/{n_splits}", end="  ")
        data = DataBundle(
            df_train=df.iloc[train_idx].copy(),
            df_val=df.iloc[val_idx].copy(),
        )
        fold_metrics.append(_run_fold(data, config))

    return {"fold_metrics": fold_metrics, "summary": _summarise(fold_metrics)}


def cross_validate_sweep(
    base_config: dict,
    param_grid: list[dict],
    n_splits: int = 5,
    random_state: int = 42,
) -> list[dict]:
    """Run cross_validate for multiple configs, sharing data loading and
    featurization within each fold where (featurizer, subsample) match.

    Returns a list of result dicts in param_grid order.
    """
    configs = [{**base_config, **overrides} for overrides in param_grid]
    df = load_full_df()

    # Exclude the held-out test set from CV folds
    test_ratio = base_config.get("test_ratio", 0.0)
    if test_ratio > 0:
        test_cut = int((1.0 - test_ratio) * len(df))
        df = df.iloc[:test_cut].copy()

    y_all = df["readmitted"].values

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_fold_metrics: list[list[dict]] = [[] for _ in configs]

    _feat_keys = ("featurizer", "subsample")

    def _feat_sig(cfg):
        return tuple(cfg.get(k) for k in _feat_keys)

    for fold, (train_idx, val_idx) in enumerate(kf.split(df, y_all)):
        print(f"\n[fold {fold + 1}/{n_splits}]")
        data = DataBundle(
            df_train=df.iloc[train_idx].copy(),
            df_val=df.iloc[val_idx].copy(),
        )

        groups: dict[tuple, list[int]] = {}
        for i, cfg in enumerate(configs):
            groups.setdefault(_feat_sig(cfg), []).append(i)

        for sig, indices in groups.items():
            features = stage_featurize(data, configs[indices[0]])
            for i in indices:
                cfg = configs[i]
                print(f"  -> {cfg.get('name', cfg.get('model', '?'))}", end="  ")
                all_fold_metrics[i].append(_run_fold(data, cfg, features=features))

    return [
        {"fold_metrics": fm, "summary": _summarise(fm)}
        for fm in all_fold_metrics
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_fold(data: DataBundle, config: dict, features: FeatureBundle | None = None) -> dict:
    if features is None:
        features = stage_featurize(data, config)
    config   = {**config, "input_dim": features.n_features}
    run_dir  = stage_train(features, config)
    return stage_evaluate(features, config, run_dir)


def _summarise(fold_metrics: list[dict]) -> dict:
    summary = {}
    for key in _SCALAR_KEYS:
        vals = [m[key] for m in fold_metrics if key in m]
        if vals:
            summary[f"{key}_mean"] = float(np.mean(vals))
            summary[f"{key}_std"]  = float(np.std(vals))
    return summary
