"""PRIMARY STRATEGY: SMC/ICT-style liquidity sweep -> structure shift -> FVG.

Only this signal module is replaced; configuration, EP-EL, false-signal filter,
Kelly sizing, regime filter, risk controls, Deriv integration, and dashboard remain
unchanged. All decisions use information available by the close of the signal candle.

Bullish setup (CALL):
  1) price sweeps below a previously confirmed swing low and closes back above it;
  2) within the next `structure_window` candles, a close breaks the latest confirmed
     swing high (bullish structure shift);
  3) within `fvg_window` candles of the shift, a bullish three-candle FVG appears:
     current low > high two candles earlier.

Bearish setup (PUT) is the mirrored condition. Swing points are only used after
right-side confirmation, so the detector does not use future candles. A setup expires
if its next stage does not appear within its fixed window. At most one signal is
emitted per candle. The contract outcome simulation and all downstream gates retain
the original bot's behaviour.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from config import Config
from indicators import Candles, zscore, efficiency_ratio, atr, percentile_rank
import layer2_false_signal as l2

# Fixed structural definitions; intentionally not added to config so no other
# configuration values or UI controls are changed.
PIVOT_LEFT = 2
PIVOT_RIGHT = 2
STRUCTURE_WINDOW = 8
FVG_WINDOW = 3
SWEEP_ATR_TOLERANCE = 0.0


@dataclass
class Signal:
    idx: int            # index of the closed candle that completed the SMC/ICT setup
    epoch: int          # open time of that candle
    direction: str      # "CALL" or "PUT"
    z: float
    er: float
    atr_pct: float
    l2_pass: bool
    exp_idx: int        # candle whose close settles the contract
    win: bool | None    # None until the expiry candle exists


def _confirmed_pivot(c: Candles, i: int, left: int, right: int) -> tuple[float | None, float | None]:
    """Return pivot high/low confirmed *at i*, centered at i-right."""
    p = i - right
    if p - left < 0 or p + right >= len(c):
        return None, None
    lo, hi = p - left, p + right + 1
    highs, lows = c.high[lo:hi], c.low[lo:hi]
    # Strict pivot to avoid treating flat equal highs/lows as multiple levels.
    ph = float(c.high[p]) if c.high[p] == np.max(highs) and np.count_nonzero(highs == c.high[p]) == 1 else None
    pl = float(c.low[p]) if c.low[p] == np.min(lows) and np.count_nonzero(lows == c.low[p]) == 1 else None
    return ph, pl


def find_signals(c: Candles, cfg: Config) -> list[Signal]:
    n = len(c)
    if n < PIVOT_LEFT + PIVOT_RIGHT + 5:
        return []

    # These features remain available to the unchanged L2, dashboard and journal.
    z = zscore(c.close, cfg.lookback)
    er = efficiency_ratio(c.close, cfg.lookback)
    a = atr(c, cfg.atr_period)

    last_swing_high: float | None = None
    last_swing_low: float | None = None
    # Pending sweep: (direction, sweep candle index, swept level)
    sweep: tuple[str, int, float] | None = None
    # Pending shift: (direction, shift candle index)
    shift: tuple[str, int] | None = None
    out: list[Signal] = []

    for i in range(n):
        # Levels known before this candle closes; newly confirmed pivots are applied
        # only after evaluating this candle, avoiding same-candle level lookahead.
        prior_high, prior_low = last_swing_high, last_swing_low

        if prior_low is not None and c.low[i] < prior_low - SWEEP_ATR_TOLERANCE * (a[i] if not np.isnan(a[i]) else 0.0) and c.close[i] > prior_low:
            sweep = ("CALL", i, prior_low)
            shift = None
        elif prior_high is not None and c.high[i] > prior_high + SWEEP_ATR_TOLERANCE * (a[i] if not np.isnan(a[i]) else 0.0) and c.close[i] < prior_high:
            sweep = ("PUT", i, prior_high)
            shift = None

        # Expire a sweep if a structure shift did not follow promptly.
        if sweep is not None and shift is None and i - sweep[1] > STRUCTURE_WINDOW:
            sweep = None
        if shift is not None and i - shift[1] > FVG_WINDOW:
            shift = None
            sweep = None

        # Structure shift must happen after the sweep, against the latest confirmed
        # opposing swing level known before this candle.
        if sweep is not None and shift is None and i > sweep[1]:
            if sweep[0] == "CALL" and prior_high is not None and c.close[i] > prior_high:
                shift = ("CALL", i)
            elif sweep[0] == "PUT" and prior_low is not None and c.close[i] < prior_low:
                shift = ("PUT", i)

        # Three-candle fair-value gap; use only candles up to current i.
        if shift is not None and i >= 2 and i >= shift[1]:
            direction, shift_idx = shift
            bullish_fvg = c.low[i] > c.high[i - 2]
            bearish_fvg = c.high[i] < c.low[i - 2]
            confirmed = (direction == "CALL" and bullish_fvg) or (direction == "PUT" and bearish_fvg)
            if confirmed:
                exp = i + cfg.hold_candles
                win = None
                if exp < n:
                    entry, exit_ = c.open[i + 1], c.close[exp]
                    win = bool(exit_ < entry) if direction == "PUT" else bool(exit_ > entry)
                zi = float(z[i]) if not np.isnan(z[i]) else 0.0
                eri = float(er[i]) if not np.isnan(er[i]) else float("nan")
                out.append(Signal(
                    idx=i,
                    epoch=int(c.epoch[i]),
                    direction=direction,
                    z=zi,
                    er=eri,
                    atr_pct=percentile_rank(a, i, cfg.regime_lookback),
                    l2_pass=l2.passes(eri, abs(zi), cfg)[0],
                    exp_idx=exp,
                    win=win,
                ))
                # One completed setup per sequence; require a fresh sweep.
                sweep = None
                shift = None

        # Update confirmed swing levels after evaluating current bar. The pivot is
        # centered PIVOT_RIGHT bars earlier and is therefore known at current close.
        ph, pl = _confirmed_pivot(c, i, PIVOT_LEFT, PIVOT_RIGHT)
        if ph is not None:
            last_swing_high = ph
        if pl is not None:
            last_swing_low = pl

    return out
