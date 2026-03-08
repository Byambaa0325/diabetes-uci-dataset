from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    f1_score,
    recall_score,
)


def evaluate(
    model,
    X,
    y,
    threshold: float | None = None,
    plot: bool = False,
    title: str = "",
    predict_batch_size: int | None = None,
) -> dict:
    """Return evaluation metrics for a binary classifier.

    Accepts a PyTorch nn.Module or an sklearn estimator.
    X and y can be torch.Tensors or numpy arrays.

    Args:
        model:     Fitted model.
        X:         Feature matrix.
        y:         True binary labels.
        threshold: Decision threshold for hard predictions. Defaults to 0.5.
                   Pass None to use the F1-optimal threshold automatically.
        plot:      If True, render precision/recall/F1 vs threshold + confusion matrix.
        title:     Optional label shown on the plot title.

    Returns:
        Dict with keys: roc_auc, threshold, f1, recall, recall_lt30, precision,
                        report, confusion_matrix, probs.

                        recall     - macro-averaged recall across both classes.
                        recall_lt30 - recall for the positive class (<30 days readmitted).
    """
    y_np = y.numpy().astype(int) if isinstance(y, torch.Tensor) else np.asarray(y, dtype=int)

    if isinstance(model, nn.Module):
        probs = _predict_proba_torch(model, X)
    else:
        probs = _predict_proba_sklearn(model, X, batch_size=predict_batch_size)

    # Resolve threshold
    if threshold is None:
        threshold = _optimal_f1_threshold(probs, y_np)

    preds = (probs >= threshold).astype(int)

    results = {
        "roc_auc":        roc_auc_score(y_np, probs),
        "threshold":      threshold,
        "f1":             f1_score(y_np, preds, zero_division=0),
        "recall":         recall_score(y_np, preds, average="macro", zero_division=0),
        "recall_lt30":    recall_score(y_np, preds, pos_label=1, average="binary", zero_division=0),
        "precision":      _precision(y_np, preds),
        "report":         classification_report(y_np, preds, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_np, preds).tolist(),
        "probs":          probs,
    }

    if plot:
        _plot(probs, y_np, results, title)

    return results


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def _optimal_f1_threshold(probs: np.ndarray, y: np.ndarray) -> float:
    """Return the threshold that maximises F1 on the given set."""
    precision, recall, thresholds = precision_recall_curve(y, probs)
    f1_scores = np.where(
        (precision + recall) == 0, 0,
        2 * precision * recall / (precision + recall + 1e-9),
    )
    return float(thresholds[np.argmax(f1_scores[:-1])])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(probs: np.ndarray, y: np.ndarray, results: dict, title: str) -> None:
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    _plot_threshold_curve(probs, y, results["threshold"], axes[0])
    _plot_pr_curve(probs, y, axes[1])
    _plot_roc_curve(probs, y, results["roc_auc"], axes[2])
    _plot_confusion_matrix(results["confusion_matrix"], axes[3])

    label = f" — {title}" if title else ""
    fig.suptitle(
        f"Evaluation{label}  |  ROC-AUC {results['roc_auc']:.4f}  "
        f"|  F1 {results['f1']:.4f}  |  threshold {results['threshold']:.3f}",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plt.show()


def _plot_threshold_curve(probs, y, chosen_threshold, ax):
    precision, recall, thresholds = precision_recall_curve(y, probs)
    f1_scores = np.where(
        (precision[:-1] + recall[:-1]) == 0, 0,
        2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9),
    )

    ax.plot(thresholds, precision[:-1], label="Precision", linewidth=1.5)
    ax.plot(thresholds, recall[:-1],    label="Recall",    linewidth=1.5)
    ax.plot(thresholds, f1_scores,      label="F1",        linewidth=2, color="#C44E52")
    ax.axvline(chosen_threshold, color="gray", linestyle="--", linewidth=1,
               label=f"threshold={chosen_threshold:.3f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs Threshold")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    sns.despine(ax=ax)


def _plot_pr_curve(probs, y, ax):
    precision, recall, _ = precision_recall_curve(y, probs)
    ax.plot(recall, precision, linewidth=2, color="#4C72B0")
    ax.axhline(y.mean(), color="gray", linestyle="--", linewidth=1,
               label=f"no-skill ({y.mean():.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    sns.despine(ax=ax)


def _plot_roc_curve(probs, y, roc_auc, ax):
    fpr, tpr, thresholds = roc_curve(y, probs)
    ax.plot(fpr, tpr, linewidth=2, color="#C44E52", label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="No skill")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    sns.despine(ax=ax)


def _plot_confusion_matrix(cm: list, ax):
    cm_arr = np.array(cm)
    sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"],
                linewidths=0.5)
    ax.set_title("Confusion Matrix")


# ---------------------------------------------------------------------------
# Private predict helpers
# ---------------------------------------------------------------------------

def _predict_proba_torch(model: nn.Module, X) -> np.ndarray:
    X_t = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X_t).squeeze()).numpy()


def _predict_proba_sklearn(model, X, batch_size: int | None = None) -> np.ndarray:
    X_np = X.numpy() if isinstance(X, torch.Tensor) else np.asarray(X)

    def _predict_chunk(chunk):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(chunk)[:, 1]
        scores = model.decision_function(chunk)
        return 1.0 / (1.0 + np.exp(-scores))

    if batch_size is None or len(X_np) <= batch_size:
        return _predict_chunk(X_np)

    return np.concatenate([
        _predict_chunk(X_np[i : i + batch_size])
        for i in range(0, len(X_np), batch_size)
    ])



def _precision(y_true, y_pred) -> float:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0
