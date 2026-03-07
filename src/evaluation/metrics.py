import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix


def evaluate(model, X, y) -> dict:
    """Return evaluation metrics for a binary classifier.

    Accepts either a PyTorch nn.Module or an sklearn estimator.
    X and y can be torch.Tensors or numpy arrays.
    """
    y_np = y.numpy().astype(int) if isinstance(y, torch.Tensor) else np.asarray(y, dtype=int)

    if isinstance(model, nn.Module):
        probs, preds = _predict_torch(model, X)
    else:
        probs, preds = _predict_sklearn(model, X)

    return {
        "roc_auc": roc_auc_score(y_np, probs),
        "report": classification_report(y_np, preds, output_dict=True),
        "confusion_matrix": confusion_matrix(y_np, preds).tolist(),
    }


def _predict_torch(model: nn.Module, X) -> tuple[np.ndarray, np.ndarray]:
    X_t = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_t).squeeze()).numpy()
    return probs, (probs >= 0.5).astype(int)


def _predict_sklearn(model, X) -> tuple[np.ndarray, np.ndarray]:
    X_np = X.numpy() if isinstance(X, torch.Tensor) else np.asarray(X)
    probs = model.predict_proba(X_np)[:, 1]
    return probs, (probs >= 0.5).astype(int)
