from __future__ import annotations
import math


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = wins / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def wilson_lower(wins: int, n: int, z: float = 1.645) -> float:
    return wilson_interval(wins, n, z)[0]


def breakeven(b: float) -> float:
    """Win probability needed for zero expected value when a win pays b per $1 and a loss costs $1."""
    return 1.0 / (1.0 + b)
