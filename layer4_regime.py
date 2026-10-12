"""LAYER 4 - volatility / regime gate. OFF by default.
Only switch it on if out-of-sample results show the strategy fails in a specific volatility regime
(run `python -m bot.backtest --compare-l4`). Regime gates are the easiest thing to overfit."""
from __future__ import annotations
from config import Config


def passes(atr_pct: float, cfg: Config) -> tuple[bool, str]:
    if not cfg.l4_enabled:
        return True, "L4: disabled"
    if atr_pct != atr_pct:
        return False, "L4: ATR percentile unavailable"
    if atr_pct < cfg.atr_pct_min:
        return False, f"L4: volatility too low (ATR pct {atr_pct:.0f} < {cfg.atr_pct_min:.0f})"
    if atr_pct > cfg.atr_pct_max:
        return False, f"L4: volatility too high (ATR pct {atr_pct:.0f} > {cfg.atr_pct_max:.0f})"
    return True, "L4: volatility in range"
