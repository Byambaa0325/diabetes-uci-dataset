"""Stage 3 — Train model and persist artefacts."""
from __future__ import annotations

import joblib
from pathlib import Path

from src.models.registry import MODEL_REGISTRY
from src.models.trainers import run_training
from src.pipeline.types import FeatureBundle

# Keys consumed by the pipeline that must not be forwarded to run_training
_PIPELINE_KEYS = {"split_ratio", "test_ratio", "subsample", "featurizer", "threshold", "plot", "cv_splits", "predict_batch_size"}


def stage_train(features: FeatureBundle, config: dict) -> Path:
    """Assemble training config, run training, save scaler to run_dir.

    Returns the run directory path.

    Config keys forwarded to run_training (everything except pipeline keys).
    PyTorch models have input_dim auto-filled from FeatureBundle.n_features.
    """
    train_cfg = {k: v for k, v in config.items() if k not in _PIPELINE_KEYS}

    train_cfg["X_train"] = features.X_train
    train_cfg["y_train"] = features.y_train
    train_cfg["X_val"]   = features.X_val
    train_cfg["y_val"]   = features.y_val

    key = train_cfg.get("model", train_cfg.get("network"))
    if MODEL_REGISTRY.get(key, {}).get("framework") == "torch":
        train_cfg.setdefault("input_dim", features.n_features)

    run_dir = run_training(train_cfg)
    joblib.dump(features.scaler, run_dir / "scaler.joblib")
    return run_dir
