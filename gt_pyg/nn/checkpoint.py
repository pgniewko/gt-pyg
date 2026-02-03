"""Checkpoint utilities for gt-pyg models."""

from typing import Dict, Any, Optional, Union
from pathlib import Path

import torch

CHECKPOINT_VERSION = 1


def save_checkpoint(
    model: torch.nn.Module,
    path: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    global_step: Optional[int] = None,
    best_metric: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save checkpoint to disk (generic utility).

    Args:
        model: PyTorch model.
        path: File path.
        config: Model config for reconstruction.
        optimizer: Optimizer state.
        scheduler: LR scheduler state.
        epoch: Current epoch.
        global_step: Training step.
        best_metric: Best metric value.
        extra: Additional data.
    """
    from datetime import datetime

    path = Path(path)
    if path.suffix != ".pt":
        path = path.with_suffix(".pt")
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "created_at": datetime.utcnow().isoformat(),
        "model_state_dict": model.state_dict(),
    }

    if config is not None:
        checkpoint["model_config"] = config
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if global_step is not None:
        checkpoint["global_step"] = global_step
    if best_metric is not None:
        checkpoint["best_metric"] = best_metric
    if extra is not None:
        checkpoint["extra"] = extra

    torch.save(checkpoint, path)


def load_checkpoint(
    path: Union[str, Path],
    map_location: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """
    Load checkpoint from disk.

    Args:
        path: Checkpoint file path.
        map_location: Device mapping.

    Returns:
        Checkpoint dict with state_dict, config, etc.
    """
    return torch.load(path, map_location=map_location, weights_only=False)


def get_checkpoint_info(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get checkpoint metadata without loading full state.

    Args:
        path: Checkpoint file path.

    Returns:
        Dict with version, created_at, config, epoch, etc. (no state_dicts).
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    info = {}
    for key in ["checkpoint_version", "created_at", "model_config", "epoch",
                "global_step", "best_metric", "frozen_status", "extra"]:
        if key in checkpoint:
            info[key] = checkpoint[key]
    return info
