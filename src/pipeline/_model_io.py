"""Shared model loading and JSON serialisation helpers used by stage_evaluate and stage_test."""
from __future__ import annotations

import numpy as np
from pathlib import Path


def load_model_from_run(run_dir: Path, config: dict, n_features: int | None = None):
    """Detect and load a model artifact from run_dir.

    Supports sklearn models (joblib) and PyTorch MLP (weights.pt).
    n_features is used as input_dim fallback when not present in config.
    """
    import torch, joblib
    from src.models.registry import MODEL_REGISTRY
    from src.networks.mlp import MLP

    key = config.get("model", config.get("network"))
    framework = MODEL_REGISTRY.get(key, {}).get("framework")

    if framework == "torch":
        model = MLP(
            input_dim=config.get("input_dim") or n_features,
            hidden_dims=config.get("hidden_dims", [64, 32]),
        )
        model.load_state_dict(torch.load(run_dir / "weights.pt", weights_only=True))
        model.eval()
        return model

    model_path = run_dir / "model.joblib"
    if model_path.exists():
        return joblib.load(model_path)

    raise FileNotFoundError(f"No model artifact found in {run_dir}")


def serialisable(metrics: dict) -> dict:
    """Return a JSON-safe copy of a metrics dict.

    Scalars → float, lists/dicts kept as-is, numpy arrays excluded (e.g. probs).
    """
    result = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            result[k] = float(v)
        elif isinstance(v, (list, dict)):
            result[k] = v
        elif isinstance(v, np.ndarray):
            pass
    return result
