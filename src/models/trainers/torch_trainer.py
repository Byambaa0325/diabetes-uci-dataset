import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from src.models.trainers.base import BaseTrainer
from src.utils.logger import log_metrics


class TorchTrainer(BaseTrainer):
    """Trains an nn.Module with Adam + BCEWithLogitsLoss."""

    def train(self, config: dict, run_dir: Path, wandb_run) -> nn.Module:
        model = self.model_cls(
            input_dim=config["input_dim"],
            hidden_dims=config["hidden_dims"],
            output_dim=config.get("output_dim", 1),
            dropout=config.get("dropout", 0.3),
        )

        X_train, y_train = _to_tensor(config["X_train"]), _to_tensor(config["y_train"])
        X_val,   y_val   = _to_tensor(config["X_val"]),   _to_tensor(config["y_val"])

        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        criterion = nn.BCEWithLogitsLoss()
        loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=config["batch_size"],
            shuffle=True,
        )

        for epoch in range(config["epochs"]):
            model.train()
            total_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = criterion(model(X_batch).squeeze(), y_batch.float())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val).squeeze(), y_val.float()).item()

            log_metrics(wandb_run, {
                "train_loss": total_loss / len(loader),
                "val_loss": val_loss,
            }, step=epoch)

        torch.save(model.state_dict(), run_dir / "weights.pt")
        return model


def _to_tensor(x) -> torch.Tensor:
    return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
