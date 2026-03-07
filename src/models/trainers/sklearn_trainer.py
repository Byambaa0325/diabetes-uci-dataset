import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import log_loss

from src.models.trainers.base import BaseTrainer
from src.utils.logger import log_metrics


class SklearnTrainer(BaseTrainer):
    """Fits an sklearn estimator and saves it with joblib."""

    def train(self, config: dict, run_dir: Path, wandb_run) -> object:
        X_train = _to_numpy(config["X_train"])
        y_train = _to_numpy(config["y_train"])
        X_val   = _to_numpy(config["X_val"])
        y_val   = _to_numpy(config["y_val"])

        model = self.model_cls(**(config.get("model_params") or {}))
        model.fit(X_train, y_train)

        log_metrics(wandb_run, {
            "train_log_loss": log_loss(y_train, model.predict_proba(X_train)[:, 1]),
            "val_log_loss":   log_loss(y_val,   model.predict_proba(X_val)[:, 1]),
        }, step=0)

        joblib.dump(model, run_dir / "model.joblib")
        return model


def _to_numpy(x) -> np.ndarray:
    import torch
    return x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
