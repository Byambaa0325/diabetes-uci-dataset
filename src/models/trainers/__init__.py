import json
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

from src.models.registry import MODEL_REGISTRY
from src.models.trainers.torch_trainer import TorchTrainer
from src.models.trainers.sklearn_trainer import SklearnTrainer
from src.utils.logger import init_run, finish_run

RUNS_DIR = Path(__file__).resolve().parents[3] / "runs"

_TRAINER_MAP = {
    "torch":   TorchTrainer,
    "sklearn": SklearnTrainer,
}


def run_training(config: dict) -> Path:
    """Orchestrate a full training run.

    Resolves the model from MODEL_REGISTRY, selects the appropriate trainer,
    saves artefacts to runs/<timestamp>_<name>/, and logs to W&B.

    Returns the run directory path.
    """
    key = config.get("model", config.get("network"))
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{key}'. Available: {list(MODEL_REGISTRY)}")

    entry = MODEL_REGISTRY[key]
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = RUNS_DIR / f"{run_id}_{config.get('name', key)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = init_run(config, run_id)

    TrainerClass = _TRAINER_MAP[entry["framework"]]
    TrainerClass(entry["cls"]).train(config, run_dir, wandb_run)

    _save_config(config, run_dir)
    finish_run(wandb_run)
    print(f"Run saved → {run_dir}")
    return run_dir


def _save_config(config: dict, run_dir: Path) -> None:
    serialisable = {
        k: v for k, v in config.items()
        if not isinstance(v, (torch.Tensor, np.ndarray))
    }
    (run_dir / "config.json").write_text(json.dumps(serialisable, indent=2))
