"""Checkpoint utilities for gt-pyg models."""

import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path

import torch

from gt_pyg import __version__

logger = logging.getLogger(__name__)

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
    from datetime import datetime, timezone

    path = Path(path)
    if path.suffix != ".pt":
        path = path.with_suffix(".pt")
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "gt_pyg_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
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
    version_check: str = "warn",
) -> Dict[str, Any]:
    """
    Load checkpoint from disk.

    Args:
        path: Checkpoint file path.
        map_location: Device mapping.
        version_check: How to handle gt-pyg version mismatches.
            ``"warn"`` (default) logs a warning, ``"error"`` raises
            :class:`RuntimeError`, ``"ignore"`` skips the check.

    Returns:
        Checkpoint dict with state_dict, config, etc.

    Raises:
        RuntimeError: If ``version_check="error"`` and the checkpoint was
            saved with a different gt-pyg version.
    """
    if version_check not in ("warn", "error", "ignore"):
        raise ValueError(
            f"version_check must be 'warn', 'error', or 'ignore', "
            f"got {version_check!r}"
        )

    # NOTE: weights_only=False is intentional — checkpoints store non-tensor
    # metadata (config dicts, version strings, etc.).  Only load files you trust.
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    if version_check != "ignore":
        saved_version = checkpoint.get("gt_pyg_version")
        if saved_version is None:
            msg = (
                f"Checkpoint '{path}' has no gt_pyg_version field; "
                f"it may have been created with an older version of gt-pyg."
            )
            if version_check == "error":
                raise RuntimeError(msg)
            logger.warning(msg)
        elif saved_version != __version__:
            msg = (
                f"Checkpoint '{path}' was saved with gt-pyg {saved_version}, "
                f"but the current version is {__version__}. "
                f"Model architecture (feature dimensions, layer structure) may "
                f"have changed between versions — weights may be incompatible."
            )
            if version_check == "error":
                raise RuntimeError(msg)
            logger.warning(msg)

    return checkpoint


def get_checkpoint_info(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get checkpoint metadata without loading full state.

    Args:
        path: Checkpoint file path.

    Returns:
        Dict with version, created_at, config, epoch, etc. (no state_dicts).
    """
    # NOTE: weights_only=False is intentional — see load_checkpoint above.
    # mmap=True avoids loading large tensor data into RAM — only metadata
    # keys are deserialized since we never access state dicts.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    info = {}
    for key in ["checkpoint_version", "gt_pyg_version", "created_at",
                "model_config", "epoch", "global_step", "best_metric",
                "extra"]:
        if key in checkpoint:
            info[key] = checkpoint[key]

    # frozen_status is stored inside extra by GraphTransformerNet.save_checkpoint
    extra = checkpoint.get("extra")
    if isinstance(extra, dict) and "frozen_status" in extra:
        info["frozen_status"] = extra["frozen_status"]

    return info
