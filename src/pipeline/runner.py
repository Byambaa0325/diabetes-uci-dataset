"""Composed pipeline — chains all four stages into a single call."""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import joblib

from src.pipeline.types import DataBundle, FeatureBundle, PipelineResult
from src.pipeline.load import stage_load
from src.pipeline.featurize import stage_featurize
from src.pipeline.run_train import stage_train
from src.pipeline.run_evaluate import stage_evaluate
from src.pipeline.run_test import stage_test

# Keys that affect which DataBundle is produced (stage_load)
_LOAD_KEYS = frozenset({"split_ratio"})
# Keys that affect which FeatureBundle is produced (stage_featurize)
_FEATURIZE_KEYS = frozenset({"featurizer", "subsample"})


def run_pipeline(config: dict) -> PipelineResult:
    """Run the full pipeline and return a PipelineResult.

    Behaviour is controlled by config:

        cv_splits (int, default None):
            None  → single 85/15 holdout split (fast, for iteration).
            int   → stratified k-fold CV (correct for final evaluation).
                    metrics dict contains mean values; result.fold_metrics
                    holds per-fold dicts; result.is_cv is True.

    All other config keys are forwarded to the appropriate stages.
    """
    cv_splits = config.get("cv_splits")
    if cv_splits:
        return _run_cv(config, n_splits=int(cv_splits))

    data     = stage_load(config)
    features = stage_featurize(data, config)
    return run_pipeline_from_features(features, config)


def run_pipeline_from_features(features: FeatureBundle, config: dict) -> PipelineResult:
    """Run train + evaluate only, using a pre-built FeatureBundle.

    Use this when sweeping model hyperparameters with fixed featurization —
    build the FeatureBundle once and pass it to each config in the sweep.
    Always performs a single-split evaluation (no CV).
    """
    t0 = time.time()

    run_dir      = stage_train(features, config)
    metrics      = stage_evaluate(features, config, run_dir)
    test_metrics = stage_test(features, config, run_dir) if features.X_test is not None else None

    elapsed = time.time() - t0
    model   = _reload_model(run_dir, config, n_features=features.n_features)

    _print_result(metrics, test_metrics, elapsed)

    return PipelineResult(
        run_dir=run_dir,
        model=model,
        metrics=metrics,
        feature_bundle=features,
        elapsed_s=elapsed,
        test_metrics=test_metrics,
    )


def sweep(base_config: dict, param_grid: list[dict]) -> list[PipelineResult]:
    """Run many configs efficiently, reusing precomputed data/features.

    Always uses single-split evaluation. For CV sweeps use cross_validate_sweep.

    Groups configs by (split_ratio, featurizer, subsample) so that
    load + featurize is shared across configs that differ only in model
    hyperparameters. Preserves param_grid ordering in the returned list.
    """
    configs = [{**base_config, **overrides} for overrides in param_grid]

    _stage_keys = sorted(_LOAD_KEYS | _FEATURIZE_KEYS)

    def _stage_sig(cfg: dict) -> tuple:
        return tuple(cfg.get(k) for k in _stage_keys)

    groups: dict[tuple, list[tuple[int, dict]]] = defaultdict(list)
    for i, cfg in enumerate(configs):
        groups[_stage_sig(cfg)].append((i, cfg))

    results: list[PipelineResult | None] = [None] * len(configs)

    for sig, group in groups.items():
        ref_cfg  = group[0][1]
        data     = stage_load(ref_cfg)
        features = stage_featurize(data, ref_cfg)

        n = len(group)
        print(f"\n[sweep] featurizer={ref_cfg.get('featurizer', 'full')}  "
              f"subsample={ref_cfg.get('subsample')}  "
              f"({n} config{'s' if n > 1 else ''})")

        for i, cfg in group:
            print(f"  -> {cfg.get('name', cfg.get('model', '?'))}")
            results[i] = run_pipeline_from_features(features, cfg)

    return results


def load_model(config: dict, run_dir: Path):
    """Reload a saved model from a completed run directory."""
    return _reload_model(run_dir, config)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_cv(config: dict, n_splits: int) -> PipelineResult:
    """Run k-fold CV and return a PipelineResult with fold data."""
    from src.pipeline.cross_validate import cross_validate
    t0 = time.time()

    cv_result = cross_validate(config, n_splits=n_splits)

    # Expose mean values under the standard metric keys so downstream code
    # reading result.metrics['roc_auc'] works without modification.
    # Std values are available as result.metrics['roc_auc_std'] etc.
    summary = cv_result["summary"]
    metrics = {k.replace("_mean", ""): v for k, v in summary.items() if k.endswith("_mean")}
    metrics.update({k: v for k, v in summary.items() if k.endswith("_std")})

    elapsed = time.time() - t0
    print(
        f"  CV({n_splits}-fold)  "
        f"ROC-AUC={metrics.get('roc_auc', 0):.4f}±{metrics.get('roc_auc_std', 0):.4f}  "
        f"F1={metrics.get('f1', 0):.4f}±{metrics.get('f1_std', 0):.4f}  "
        f"({elapsed:.1f}s)"
    )

    return PipelineResult(
        run_dir=None,
        model=None,
        metrics=metrics,
        feature_bundle=None,
        elapsed_s=elapsed,
        fold_metrics=cv_result["fold_metrics"],
    )


def _print_result(metrics: dict, test_metrics: dict | None, elapsed: float) -> None:
    val_line = (
        f"  [val]  ROC-AUC={metrics.get('roc_auc', 0):.4f}  "
        f"F1={metrics.get('f1', 0):.4f}  "
        f"Recall={metrics.get('recall', 0):.4f}  "
        f"Recall(<30)={metrics.get('recall_pos', 0):.4f}  "
        f"({elapsed:.1f}s)"
    )
    print(val_line)
    if test_metrics:
        print(
            f"  [test] ROC-AUC={test_metrics.get('roc_auc', 0):.4f}  "
            f"F1={test_metrics.get('f1', 0):.4f}  "
            f"Recall={test_metrics.get('recall', 0):.4f}  "
            f"Recall(<30)={test_metrics.get('recall_pos', 0):.4f}"
        )


def _reload_model(run_dir: Path, config: dict, n_features: int | None = None):
    import torch
    from src.models.registry import MODEL_REGISTRY
    from src.networks.mlp import MLP

    key = config.get("model", config.get("network"))
    framework = MODEL_REGISTRY.get(key, {}).get("framework")

    if framework == "torch":
        weights_path = run_dir / "weights.pt"
        model = MLP(
            input_dim=config.get("input_dim") or n_features,
            hidden_dims=config.get("hidden_dims", [64, 32]),
        )
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        model.eval()
        return model

    model_path = run_dir / "model.joblib"
    if model_path.exists():
        return joblib.load(model_path)

    return None
