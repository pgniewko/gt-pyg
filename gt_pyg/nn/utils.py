from numbers import Real
from typing import Sequence

from torch import nn


VALID_AGGREGATORS = frozenset(
    {
        "sum",
        "add",
        "mean",
        "min",
        "max",
        "mul",
        "var",
        "std",
        "softmax",
        "powermean",
        "median",
    }
)

# Trainable readouts only valid for graph-level pooling, not message passing.
VALID_POOL_AGGREGATORS = VALID_AGGREGATORS | {"attn"}


def make_norm(norm: str | None, dim: int) -> nn.Module:
    """Create a BatchNorm1d ("bn"), LayerNorm ("ln"), or Identity ("none") module of size ``dim``"""
    key = norm.lower() if norm is not None else None
    if key in ("bn", "batchnorm", "batch_norm"):
        return nn.BatchNorm1d(dim)
    if key in ("ln", "layernorm", "layer_norm"):
        return nn.LayerNorm(dim)
    if key in ("none", None, "identity"):
        return nn.Identity()
    raise ValueError(f"Unknown norm type: {norm}")


def validate_dropout(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number in [0, 1), got {value!r}")
    if not 0.0 <= float(value) < 1.0:
        raise ValueError(f"{name} must be in [0, 1), got {value}")


def validate_aggregators(
    name: str,
    aggregators: Sequence[str],
    valid: frozenset = VALID_AGGREGATORS,
) -> None:
    if isinstance(aggregators, (str, bytes)) or not isinstance(aggregators, (list, tuple)):
        raise ValueError(f"{name} must be a non-empty list or tuple of aggregator names")
    if len(aggregators) == 0:
        raise ValueError(f"{name} must contain at least one aggregator")

    invalid = []
    for aggregator in aggregators:
        if not isinstance(aggregator, str):
            raise ValueError(f"{name} entries must be strings, got {aggregator!r}")
        if aggregator == "":
            raise ValueError(f"{name} entries must be non-empty strings")
        if aggregator not in valid:
            invalid.append(aggregator)

    if invalid:
        valid_names = ", ".join(sorted(valid))
        raise ValueError(
            f"{name} contains unsupported aggregators {invalid!r}; "
            f"valid aggregators are: {valid_names}"
        )


def validate_num_gt_layers(num_gt_layers: int) -> None:
    if isinstance(num_gt_layers, bool) or not isinstance(num_gt_layers, int):
        raise ValueError(
            "num_gt_layers must be a non-negative integer, "
            f"got {num_gt_layers!r}"
        )
    if num_gt_layers < 0:
        raise ValueError(f"num_gt_layers must be non-negative, got {num_gt_layers}")
