"""Stage 4 — Load saved model and evaluate on the validation set."""
from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.metrics import evaluate
from src.pipeline.types import FeatureBundle
from src.pipeline._model_io import load_model_from_run, serialisable



def stage_evaluate(features: FeatureBundle, config: dict, run_dir: Path) -> dict:
    """Load saved model from run_dir and evaluate on validation data.

    Config keys:
        threshold (float | None): decision threshold; None = F1-optimal.
        plot      (bool): whether to render evaluation plots.
        name      (str):  used as plot title.
    """
    model = load_model_from_run(run_dir, config, n_features=features.n_features)
    threshold = config.get("threshold")
    plot      = config.get("plot", False)

    metrics = evaluate(
        model,
        features.X_val,
        features.y_val,
        threshold=threshold,
        plot=plot,
        title=config.get("name", ""),
        predict_batch_size=config.get("predict_batch_size"),
    )

    (run_dir / "metrics.json").write_text(
        json.dumps(serialisable(metrics), indent=2)
    )
    return metrics


