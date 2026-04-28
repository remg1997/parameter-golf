"""Minimal wandb integration for parameter-golf experiments.

Env vars:
    WANDB_API_KEY      required 
    WANDB_PROJECT      defaults to "parameter-golf"
    WANDB_DISABLED=1   skips all wandb calls (useful for smoke tests)
"""
import os

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def _is_main_rank() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _is_enabled() -> bool:
    return _WANDB_AVAILABLE and not os.environ.get("WANDB_DISABLED") and _is_main_rank()


def init_wandb(hparams_dict: dict, run_name: str | None = None, tags: list[str] | None = None):
    if not _is_enabled():
        return None
    return wandb.init(
        project=os.environ.get("WANDB_PROJECT", "parameter-golf"),
        entity=os.environ.get("WANDB_ENTITY"),
        name=run_name or os.environ.get("RUN_ID"),
        group=os.environ.get("WANDB_GROUP"),
        config=hparams_dict,
        tags=tags or [],
        reinit=True,
    )


def log_metrics(metrics: dict, step: int | None = None) -> None:
    if not _is_enabled() or wandb.run is None:
        return
    wandb.log(metrics, step=step)


def finish_wandb() -> None:
    if _is_enabled() and wandb.run is not None:
        wandb.finish()
