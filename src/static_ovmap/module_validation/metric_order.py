"""Protocol metric ties followed by exact cost/complexity tie-breaks."""

from collections.abc import Callable, Iterable, Sequence
from typing import TypeVar

T = TypeVar("T")
TOLERANCE = 1e-10


def metric_max(
    values: Iterable[T], *, key: Callable[[T], Sequence[float]], metrics: int = 2
) -> T:
    candidates = [(value, tuple(key(value))) for value in values]
    if not candidates:
        raise ValueError("metric comparison requires candidates")
    for index in range(len(candidates[0][1])):
        best = max(row[index] for _, row in candidates)
        tolerance = TOLERANCE if index < metrics else 0.0
        candidates = [
            (value, row) for value, row in candidates if row[index] >= best - tolerance
        ]
    return candidates[0][0]
