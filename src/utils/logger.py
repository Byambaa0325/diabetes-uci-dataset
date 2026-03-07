"""Thin wrapper around wandb so the rest of the codebase stays logger-agnostic."""
from typing import Optional
import wandb


def init_run(config: dict, run_id: str) -> Optional[wandb.sdk.wandb_run.Run]:
    """Initialise a wandb run. Returns the run object (or None if wandb is disabled)."""
    loggable = {k: v for k, v in config.items() if not hasattr(v, "__len__") or isinstance(v, (str, list))}
    return wandb.init(
        project=config.get("wandb_project", "applied-ai-coursework"),
        name=run_id,
        config=loggable,
        reinit=True,
    )


def log_metrics(run, metrics: dict, step: int) -> None:
    if run is not None:
        run.log(metrics, step=step)


def finish_run(run) -> None:
    if run is not None:
        run.finish()
