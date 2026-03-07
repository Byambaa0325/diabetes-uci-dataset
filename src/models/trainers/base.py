from abc import ABC, abstractmethod
from pathlib import Path


class BaseTrainer(ABC):
    """Common interface for all trainers.

    Each concrete trainer receives the model class at construction and
    implements `train`, which runs the full fit cycle, persists artefacts
    to `run_dir`, and returns the fitted model.
    """

    def __init__(self, model_cls):
        self.model_cls = model_cls

    @abstractmethod
    def train(self, config: dict, run_dir: Path, wandb_run) -> object:
        """Fit the model, save artefacts to run_dir, return the fitted model."""
        ...
