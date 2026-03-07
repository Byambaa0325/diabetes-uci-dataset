"""Bias-variance decomposition via bootstrap resampling (Kohavi-Wolpert, 0-1 loss).

For each bootstrap iteration a fresh model is trained on a resample of the
training set. For each test point we then compute:

  - **bias²**: does the majority-vote prediction disagree with the true label?
  - **variance**: how often do individual predictions disagree with the majority vote?
  - **total error**: average 0-1 loss across all bootstrap models

Relationship:  total_error ≈ bias² + variance  (noise is irreducible and not separated here)
"""

from __future__ import annotations
import numpy as np
from typing import Callable


def bias_variance_decomp(
    model_fn: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_bootstrap: int = 200,
    random_state: int = 42,
) -> dict:
    """Bootstrap bias-variance decomposition for binary classification.

    Args:
        model_fn:     Zero-argument callable that returns a fresh, unfitted
                      sklearn-compatible estimator (has .fit / .predict).
        X_train:      Training features (numpy array).
        y_train:      Training labels (numpy array, binary int).
        X_test:       Test features.
        y_test:       Test labels.
        n_bootstrap:  Number of bootstrap resamples.
        random_state: RNG seed for reproducibility.

    Returns:
        Dict with keys: bias_sq, variance, total_error, all_preds (n_bootstrap × n_test).
    """
    rng = np.random.default_rng(random_state)
    n_train = len(X_train)
    all_preds = np.empty((n_bootstrap, len(X_test)), dtype=np.int8)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n_train, size=n_train)
        model = model_fn()
        model.fit(X_train[idx], y_train[idx])
        all_preds[i] = model.predict(X_test).astype(np.int8)

    # Majority vote across bootstrap models
    majority = (all_preds.mean(axis=0) >= 0.5).astype(np.int8)

    bias_sq    = float(np.mean(majority != y_test))
    variance   = float(np.mean(all_preds != majority))
    total_error = float(np.mean(all_preds != y_test))

    return {
        "bias_sq":     bias_sq,
        "variance":    variance,
        "total_error": total_error,
        "all_preds":   all_preds,
    }


def sweep(
    model_fn_factory: Callable,
    param_values: list,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_bootstrap: int = 200,
    random_state: int = 42,
) -> list[dict]:
    """Run bias_variance_decomp across a range of complexity values.

    Args:
        model_fn_factory: Callable(param) → model_fn (zero-arg callable).
        param_values:     List of complexity parameter values to sweep.

    Returns:
        List of result dicts (one per param value), each including the param value.
    """
    results = []
    for val in param_values:
        result = bias_variance_decomp(
            model_fn_factory(val),
            X_train, y_train, X_test, y_test,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        result["param"] = val
        results.append(result)
    return results
