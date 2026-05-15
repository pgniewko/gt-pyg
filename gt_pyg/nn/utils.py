from numbers import Real
from typing import Sequence


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


def validate_dropout(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number in [0, 1), got {value!r}")
    if not 0.0 <= float(value) < 1.0:
        raise ValueError(f"{name} must be in [0, 1), got {value}")


def validate_aggregators(name: str, aggregators: Sequence[str]) -> None:
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
        if aggregator not in VALID_AGGREGATORS:
            invalid.append(aggregator)

    if invalid:
        valid = ", ".join(sorted(VALID_AGGREGATORS))
        raise ValueError(
            f"{name} contains unsupported aggregators {invalid!r}; "
            f"valid aggregators are: {valid}"
        )


def validate_num_gt_layers(num_gt_layers: int) -> None:
    if isinstance(num_gt_layers, bool) or not isinstance(num_gt_layers, int):
        raise ValueError(
            "num_gt_layers must be a non-negative integer, "
            f"got {num_gt_layers!r}"
        )
    if num_gt_layers < 0:
        raise ValueError(f"num_gt_layers must be non-negative, got {num_gt_layers}")
