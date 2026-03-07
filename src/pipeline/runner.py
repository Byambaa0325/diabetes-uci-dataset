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

# Keys that affect which DataBundle is produced (stage_load)
_LOAD_KEYS = frozenset({"split_ratio"})
# Keys that affect which FeatureBundle is produced (stage_featurize)
_FEATURIZE_KEYS = frozenset({"featurizer", "subsample"})


def run_pipeline(config: dict) -> PipelineResult:
    """Run all four stages in sequence and return a PipelineResult.

    Stages:
        load      -> DataBundle
        featurize -> FeatureBundle
        train     -> run_dir (Path)
        evaluate  -> metrics (dict)
    """
    data     = stage_load(config)
    features = stage_featurize(data, config)
    return run_pipeline_from_features(features, config)


def run_pipeline_from_features(features: FeatureBundle, config: dict) -> PipelineResult:
    """Run train + evaluate stages only, using a pre-built FeatureBundle.

    Use this when sweeping model hyperparameters with fixed featurization —
    build the FeatureBundle once and pass it to each config in the sweep.
    """
    t0 = time.time()

    run_dir = stage_train(features, config)
    metrics = stage_evaluate(features, config, run_dir)

    elapsed = time.time() - t0
    model   = _reload_model(run_dir, config, n_features=features.n_features)

    print(
        f"  ROC-AUC={metrics.get('roc_auc', 0):.4f}  "
        f"F1={metrics.get('f1', 0):.4f}  "
        f"Recall={metrics.get('recall', 0):.4f}  "
        f"Recall(<30)={metrics.get('recall_lt30', 0):.4f}  "
        f"({elapsed:.1f}s)"
    )

    return PipelineResult(
        run_dir=run_dir,
        model=model,
        metrics=metrics,
        feature_bundle=features,
        elapsed_s=elapsed,
    )


def sweep(base_config: dict, param_grid: list[dict]) -> list[PipelineResult]:
    """Run many configs efficiently, reusing precomputed data/features.

    Groups configs by (split_ratio, featurizer, subsample) so that
    load + featurize is shared across configs that differ only in model
    hyperparameters. Preserves param_grid ordering in the returned list.

    Args:
        base_config: Shared config applied as defaults to all runs.
        param_grid:  List of override dicts, one per run.  Each dict is
                     merged over base_config, so only differing keys need
                     to be specified.

    Returns:
        List of PipelineResult in the same order as param_grid.

    Example::

        results = sweep(
            base_config={"model": "logistic_regression", "featurizer": "full"},
            param_grid=[{"model_params": {"C": c}} for c in [0.01, 0.1, 1.0, 10.0]],
        )
    """
    configs = [{**base_config, **overrides} for overrides in param_grid]

    # Group by the keys that determine which (data, features) pair to use.
    # Sorted key names give a deterministic, hashable signature.
    _stage_keys = sorted(_LOAD_KEYS | _FEATURIZE_KEYS)

    def _stage_sig(cfg: dict) -> tuple:
        return tuple(cfg.get(k) for k in _stage_keys)

    # Collect groups while preserving param_grid insertion order.
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
