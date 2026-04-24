"""
Shared signal quality scoring for setup-based entries.
"""
from typing import List, Tuple


def quality_score(
    direction: str,
    sweep_present: bool,
    body_ratio: float,
    tick_ratio: float,
    tick_velocity_increasing: bool,
    m1_aligned: bool,
    m5_aligned: bool,
    compression_ok: bool,
    tick_ratio_threshold: float = 0.65,
) -> Tuple[float, List[str]]:
    score = 0.0
    reasons: List[str] = []

    if sweep_present:
        score += 0.25
        reasons.append("Liquidity sweep present")

    if body_ratio >= 0.5:
        score += 0.20
        reasons.append(f"Strong displacement ({body_ratio:.0%})")

    short_threshold = 1.0 - tick_ratio_threshold
    tick_ok = (
        direction == "LONG" and tick_ratio >= tick_ratio_threshold
    ) or (
        direction == "SHORT" and tick_ratio <= short_threshold
    )
    if tick_ok:
        score += 0.20
        reasons.append(f"Tick ratio {tick_ratio:.0%}")

    if tick_velocity_increasing:
        score += 0.15
        reasons.append("Tick velocity increasing")

    if m1_aligned and m5_aligned:
        score += 0.10
        reasons.append("M1+M5 aligned")

    if compression_ok:
        score += 0.10
        reasons.append("Compression valid")

    return round(min(1.0, score), 3), reasons
