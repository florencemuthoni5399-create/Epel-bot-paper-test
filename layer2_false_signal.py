"""LAYER 2 - false-signal filter.
Mean reversion fails when price is *trending* or *breaking out* rather than stretching and snapping back.
Two simple, pre-set checks (kept deliberately few so there is little to overfit):
  * efficiency ratio over the lookback > er_max  -> trending market, skip
  * |z| > z_abs_max                              -> breakout / news spike, skip
Whether this filter actually helps must be confirmed out-of-sample (see backtest.py)."""
from __future__ import annotations
from config import Config


def passes(er: float, z_abs: float, cfg: Config) -> tuple[bool, str]:
    if er != er:  # NaN
        return False, "L2: efficiency ratio unavailable"
    if er > cfg.er_max:
        return False, f"L2: trending (ER {er:.2f} > {cfg.er_max:.2f})"
    if z_abs > cfg.z_abs_max:
        return False, f"L2: breakout (|z| {z_abs:.2f} > {cfg.z_abs_max:.2f})"
    return True, "L2: signal looks like a stretch, not a trend"
