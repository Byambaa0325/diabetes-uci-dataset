"""Stage 4 — Load saved model and evaluate on the validation set."""
from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.metrics import evaluate
from src.pipeline.types import FeatureBundle


def stage_evaluate(features: FeatureBundle, config: dict, run_dir: Path) -> dict:
    """Load saved model from run_dir and evaluate on validation data.

    Config keys:
        threshold (float | None): decision threshold; None = F1-optimal.
        plot      (bool): whether to render evaluation plots.
        name      (str):  used as plot title.
    """
    model = _load_model_from_run(run_dir, config, n_features=features.n_features)
    threshold = config.get("threshold")
    plot      = config.get("plot", False)

    metrics = evaluate(
        model,
        features.X_val,
        features.y_val,
        threshold=threshold,
        plot=plot,
        title=config.get("name", ""),
    )

    (run_dir / "metrics.json").write_text(
        json.dumps(_serialisable(metrics), indent=2)
    )
    return metrics


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _serialisable(metrics: dict) -> dict:
    """Return a JSON-safe version of the metrics dict.

    Scalars → float, lists/dicts kept as-is, numpy arrays excluded (probs).
    """
    import numpy as np
    result = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            result[k] = float(v)
        elif isinstance(v, (list, dict)):
            result[k] = v
        elif isinstance(v, np.ndarray):
            pass  # exclude raw arrays (e.g. probs)
    return result


def _load_model_from_run(run_dir: Path, config: dict, n_features: int | None = None):
    """Detect and load a model artifact from run_dir."""
    import torch, joblib
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

    raise FileNotFoundError(f"No model artifact found in {run_dir}")
