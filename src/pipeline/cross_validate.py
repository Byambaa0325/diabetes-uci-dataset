"""K-fold cross-validation for the pipeline.

Featurization (scaler fit) is performed independently per fold so no
information from the validation fold contaminates the training fold.

Called internally by run_pipeline when config contains 'cv_splits'.
Also available directly for explicit CV workflows.

Public API
----------
cross_validate(config, n_splits, random_state, checkpoint_dir)
    -> {'fold_metrics': [...], 'summary': {'roc_auc_mean': ..., ...}}

cross_validate_sweep(base_config, param_grid, n_splits, random_state, checkpoint_dir)
    -> list of cross_validate results, one per param_grid entry.

Checkpointing
-------------
Pass checkpoint_dir to save fold results after each fold completes.
On re-run with the same checkpoint_dir, completed folds are loaded from
disk and skipped — only remaining folds are computed. Safe to interrupt
and resume any number of times.
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from src.pipeline.load import load_full_df
from src.pipeline.types import DataBundle, FeatureBundle
from src.pipeline.featurize import stage_featurize
from src.pipeline.run_train import stage_train
from src.pipeline.run_evaluate import stage_evaluate

_SCALAR_KEYS = ("roc_auc", "f1", "recall", "recall_pos", "precision", "threshold")


def cross_validate(
    config: dict,
    n_splits: int = 5,
    random_state: int = 42,
    checkpoint_dir: str | Path | None = None,
) -> dict:
    """Stratified k-fold cross-validation over the full dataset.

    Args:
        config:         Pipeline config dict (same format as run_pipeline).
        n_splits:       Number of folds.
        random_state:   Seed for fold splitting.
        checkpoint_dir: Directory to save/load per-fold results.
                        Completed folds are skipped on re-run.

    Returns:
        Dict with:
            fold_metrics  — list of metric dicts, one per fold.
            summary       — {metric_mean, metric_std} for each scalar metric.
    """
    cpdir = Path(checkpoint_dir) if checkpoint_dir else None
    if cpdir:
        cpdir.mkdir(parents=True, exist_ok=True)

    df = load_full_df()

    test_ratio = config.get("test_ratio", 0.0)
    if test_ratio > 0:
        test_cut = int((1.0 - test_ratio) * len(df))
        df = df.iloc[:test_cut].copy()

    y_all = df["readmitted"].values
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df, y_all)):
        cp_path = cpdir / f"fold_{fold + 1}.json" if cpdir else None

        if cp_path and cp_path.exists():
            metrics = json.loads(cp_path.read_text())
            fold_metrics.append(metrics)
            print(f"  fold {fold + 1}/{n_splits}  [resumed from checkpoint]")
            continue

        print(f"  fold {fold + 1}/{n_splits}", end="  ")
        data = DataBundle(
            df_train=df.iloc[train_idx].copy(),
            df_val=df.iloc[val_idx].copy(),
        )
        metrics = _run_fold(data, config)
        fold_metrics.append(metrics)

        if cp_path:
            cp_path.write_text(json.dumps(_serialisable_metrics(metrics), indent=2))
            print(f"  [saved]", end="")
        print()

    return {"fold_metrics": fold_metrics, "summary": _summarise(fold_metrics)}


def cross_validate_sweep(
    base_config: dict,
    param_grid: list[dict],
    n_splits: int = 5,
    random_state: int = 42,
    checkpoint_dir: str | Path | None = None,
) -> list[dict]:
    """Run cross_validate for multiple configs, sharing data loading and
    featurization within each fold where (featurizer, subsample) match.

    Args:
        checkpoint_dir: Directory to save/load per-fold-per-config results.
                        File names are fold_{n}_config_{i}.json.

    Returns a list of result dicts in param_grid order.
    """
    cpdir = Path(checkpoint_dir) if checkpoint_dir else None
    if cpdir:
        cpdir.mkdir(parents=True, exist_ok=True)

    configs = [{**base_config, **overrides} for overrides in param_grid]
    df = load_full_df()

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
            # Check if all configs in this group are already checkpointed
            pending = [
                i for i in indices
                if not (cpdir and (cpdir / f"fold_{fold+1}_config_{i}.json").exists())
            ]

            features = None
            if pending:
                features = stage_featurize(data, configs[indices[0]])

            for i in indices:
                cp_path = cpdir / f"fold_{fold+1}_config_{i}.json" if cpdir else None

                if cp_path and cp_path.exists():
                    metrics = json.loads(cp_path.read_text())
                    all_fold_metrics[i].append(metrics)
                    print(f"  -> {configs[i].get('name', '?')}  [resumed]")
                    continue

                cfg = configs[i]
                print(f"  -> {cfg.get('name', cfg.get('model', '?'))}", end="  ")
                metrics = _run_fold(data, cfg, features=features)
                all_fold_metrics[i].append(metrics)

                if cp_path:
                    cp_path.write_text(json.dumps(_serialisable_metrics(metrics), indent=2))

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
    config  = {**config, "input_dim": features.n_features}
    run_dir = stage_train(features, config)
    return stage_evaluate(features, config, run_dir)


def _summarise(fold_metrics: list[dict]) -> dict:
    summary = {}
    for key in _SCALAR_KEYS:
        vals = [m[key] for m in fold_metrics if key in m]
        if vals:
            summary[f"{key}_mean"] = float(np.mean(vals))
            summary[f"{key}_std"]  = float(np.std(vals))
    return summary


def _serialisable_metrics(metrics: dict) -> dict:
    """Strip non-JSON-serialisable values (numpy arrays) from a metrics dict."""
    result = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            result[k] = float(v)
        elif isinstance(v, (list, dict)):
            result[k] = v
        elif isinstance(v, np.ndarray):
            pass
    return result
