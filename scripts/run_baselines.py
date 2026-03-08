"""Run all baseline models and print a comparison table.

Usage:
    python scripts/run_baselines.py
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import sweep

BASE_CONFIG = {
    "wandb_project": "applied-ai-coursework",
}

EXPERIMENTS = [
    {
        "name":         "logistic_regression",
        "model":        "logistic_regression",
        "model_params": {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"},
    },
    {
        "name":         "random_forest",
        "model":        "random_forest",
        "model_params": {
            "n_estimators": 200, "max_depth": 10,
            "class_weight": "balanced", "n_jobs": -1, "random_state": 42,
        },
    },
    {
        "name":         "gradient_boosting",
        "model":        "gradient_boosting",
        "model_params": {
            "n_estimators": 200, "max_depth": 4,
            "learning_rate": 0.05, "subsample": 0.8, "random_state": 42,
        },
    },
    {
        "name":         "linear_svm",
        "model":        "linear_svm",
        "subsample":    15_000,
        "model_params": {
            "C": 1.0, "max_iter": 2000, "class_weight": "balanced",
        },
    },
    {
        "name":         "svm_rbf",
        "model":        "svm",
        "subsample":    15_000,
        "model_params": {
            "kernel": "rbf", "C": 1.0, "gamma": "scale",
            "probability": True, "class_weight": "balanced",
        },
    },
    {
        "name":         "mlp",
        "model":        "mlp",
        "hidden_dims":  [256, 128, 64],
        "dropout":      0.3,
        "lr":           1e-3,
        "epochs":       100,
        "patience":     10,
        "batch_size":   256,
    },
]

results = sweep(BASE_CONFIG, EXPERIMENTS)

rows = []
for cfg, result in zip(EXPERIMENTS, results):
    m = result.metrics
    rows.append({
        "model":     cfg["name"],
        "ROC-AUC":   round(m["roc_auc"], 4),
        "F1":        round(m["f1"], 4),
        "Recall":      round(m["recall"], 4),
        "Recall(<30)": round(m["recall_lt30"], 4),
        "Precision": round(m["precision"], 4),
        "Threshold": round(m["threshold"], 3),
        "Time(s)":   round(result.elapsed_s, 1),
        "run_dir":   result.run_dir.name,
    })

print("\n" + "=" * 80)
print("BASELINE RESULTS")
print("=" * 80)
summary = pd.DataFrame(rows).set_index("model")
print(summary.to_string())
