"""Stage 5 — Evaluate the trained model on the held-out test set."""
from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.metrics import evaluate
from src.pipeline.types import FeatureBundle
from src.pipeline._model_io import load_model_from_run, serialisable



def stage_test(features: FeatureBundle, config: dict, run_dir: Path) -> dict:
    """Evaluate the saved model on the test set.

    Only call this after stage_train has completed. Requires test_ratio > 0
    in the config so that features.X_test / y_test are populated.

    Config keys (same as stage_evaluate):
        threshold (float | None): decision threshold; None = F1-optimal.
        plot      (bool): whether to render evaluation plots.
        name      (str):  used as plot title.

    Returns:
        Metrics dict (same schema as stage_evaluate).
    """
    if features.X_test is None:
        raise ValueError(
            "No test set found in FeatureBundle. "
            "Set test_ratio > 0 in config to enable test evaluation."
        )

    model = load_model_from_run(run_dir, config, n_features=features.n_features)

    metrics = evaluate(
        model,
        features.X_test,
        features.y_test,
        threshold=config.get("threshold"),
        plot=config.get("plot", False),
        title=f"{config.get('name', '')} [TEST]",
        predict_batch_size=config.get("predict_batch_size"),
    )

    (run_dir / "test_metrics.json").write_text(
        json.dumps(serialisable(metrics), indent=2)
    )
    return metrics

