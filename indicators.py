from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


@dataclass
class Candles:
    epoch: np.ndarray   # candle OPEN time, unix seconds
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray

    def __len__(self) -> int:
        return len(self.epoch)


def zscore(close: np.ndarray, n: int) -> np.ndarray:
    """(close - SMA_n) / population std_n, including the current candle (standard Bollinger)."""
    out = np.full(len(close), np.nan)
    if len(close) < n:
        return out
    w = sliding_window_view(close, n)
    mean = w.mean(axis=1)
    std = w.std(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (close[n - 1:] - mean) / std
    z[std == 0] = np.nan
    out[n - 1:] = z
    return out


def efficiency_ratio(close: np.ndarray, n: int) -> np.ndarray:
    """Kaufman efficiency ratio: |net move| / total path length over n candles. ~0 = choppy, ~1 = trend."""
    out = np.full(len(close), np.nan)
    if len(close) <= n:
        return out
    d = np.abs(np.diff(close))
    cs = np.concatenate([[0.0], np.cumsum(d)])
    path = cs[n:] - cs[:-n]
    net = np.abs(close[n:] - close[:-n])
    with np.errstate(divide="ignore", invalid="ignore"):
        er = net / path
    er[path == 0] = np.nan
    out[n:] = er
    return out


def atr(c: Candles, period: int) -> np.ndarray:
    prev_close = np.concatenate([[c.close[0]], c.close[:-1]])
    tr = np.maximum.reduce([c.high - c.low, np.abs(c.high - prev_close), np.abs(c.low - prev_close)])
    out = np.full(len(tr), np.nan)
    if len(tr) >= period:
        cs = np.concatenate([[0.0], np.cumsum(tr)])
        out[period - 1:] = (cs[period:] - cs[:-period]) / period
    return out


def percentile_rank(series: np.ndarray, i: int, lookback: int) -> float:
    """Percentile (0-100) of series[i] within the previous `lookback` values (inclusive)."""
    lo = max(0, i - lookback + 1)
    w = series[lo:i + 1]
    w = w[~np.isnan(w)]
    if len(w) < 20 or np.isnan(series[i]):
        return float("nan")
    return 100.0 * float(np.mean(w <= series[i]))
