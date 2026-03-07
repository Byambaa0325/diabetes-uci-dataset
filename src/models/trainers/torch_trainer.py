import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from src.models.trainers.base import BaseTrainer
from src.utils.logger import log_metrics


class TorchTrainer(BaseTrainer):
    """Trains an nn.Module with Adam + BCEWithLogitsLoss.

    Early stopping config keys (all optional):
        patience (int):   Epochs to wait for val_loss improvement. Default 10.
                          Set to 0 to disable early stopping.
        min_delta (float): Minimum improvement to count as progress. Default 1e-4.
    """

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

        patience  = config.get("patience", 10)
        min_delta = config.get("min_delta", 1e-4)
        stopper   = _EarlyStopping(patience, min_delta, run_dir / "weights.pt")

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
                "val_loss":   val_loss,
            }, step=epoch)

            if patience > 0:
                stopper.step(val_loss, model)
                if stopper.should_stop:
                    print(f"Early stopping at epoch {epoch + 1}  "
                          f"(best val_loss {stopper.best_loss:.6f})")
                    break
        else:
            # No early stopping triggered — save final weights
            torch.save(model.state_dict(), run_dir / "weights.pt")

        if patience > 0:
            # Restore best weights regardless of whether we stopped early
            model.load_state_dict(torch.load(
                run_dir / "weights.pt", weights_only=True
            ))

        return model


# ---------------------------------------------------------------------------

class _EarlyStopping:
    """Saves best weights and signals when training should stop."""

    def __init__(self, patience: int, min_delta: float, checkpoint_path: Path):
        self.patience         = patience
        self.min_delta        = min_delta
        self.checkpoint_path  = checkpoint_path
        self.best_loss        = float("inf")
        self.epochs_no_improve = 0
        self.should_stop      = False

    def step(self, val_loss: float, model: nn.Module) -> None:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss         = val_loss
            self.epochs_no_improve = 0
            torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.should_stop = True


def _to_tensor(x) -> torch.Tensor:
    return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
