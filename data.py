from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from indicators import Candles


def save_csv(c: Candles, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "open", "high", "low", "close"])
        for row in zip(c.epoch, c.open, c.high, c.low, c.close):
            w.writerow(row)


def load_csv(path: str) -> Candles:
    rows = list(csv.DictReader(open(path)))
    return from_rows(rows)


def from_rows(rows: list[dict]) -> Candles:
    """rows: dicts with epoch/open/high/low/close. Sorted, de-duplicated by epoch."""
    by_epoch = {int(float(r["epoch"])): r for r in rows}
    ks = sorted(by_epoch)
    g = lambda k: np.array([float(by_epoch[e][k]) for e in ks])
    return Candles(np.array(ks, dtype=np.int64), g("open"), g("high"), g("low"), g("close"))


def merge(old: Candles | None, new: Candles) -> Candles:
    if old is None:
        return new
    rows = [dict(epoch=e, open=o, high=h, low=l, close=cl) for e, o, h, l, cl in zip(old.epoch, old.open, old.high, old.low, old.close)]
    rows += [dict(epoch=e, open=o, high=h, low=l, close=cl) for e, o, h, l, cl in zip(new.epoch, new.open, new.high, new.low, new.close)]
    return from_rows(rows)
