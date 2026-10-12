"""Decision engine. Runtime order:
   risk limits -> L2 false-signal filter -> (get live payout) -> L1 EP-EL gate -> L4 regime gate -> L3 sizing.
Sizing is last on purpose: Kelly sizes an eligible trade, it never turns a rejected bet into an acceptable one.
The probability estimate comes ONLY from signals whose outcome was already known at decision time."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from config import Config
from strategy import Signal
from risk import RiskManager
import layer1_epel as l1
import layer3_kelly as l3
import layer4_regime as l4


@dataclass
class Decision:
    action: str                     # "TRADE" or "SKIP"
    reason: str
    l2_pass: bool = True
    n: int = 0
    p_hat: float = float("nan")
    p_cons: float = float("nan")
    breakeven: float = float("nan")
    payout_ratio: float = float("nan")
    ep: float = float("nan")
    el: float = float("nan")
    ratio: float = float("nan")
    stake: float = 0.0
    notes: list[str] = field(default_factory=list)


class Population:
    """Resolved signals, thinned so consecutive entries are >= spacing candles apart (what a one-position bot could
    actually have traded). Greedy forward thinning is independent of the future, so it is safe to precompute."""

    def __init__(self, signals: list[Signal], spacing: int, only_l2: bool):
        idx, exp, win = [], [], []
        last = -10**9
        for s in signals:
            if s.win is None or (only_l2 and not s.l2_pass):
                continue
            if s.idx >= last + spacing:
                idx.append(s.idx); exp.append(s.exp_idx); win.append(1 if s.win else 0)
                last = s.idx
        self.exp = np.array(exp, dtype=int)
        self.win = np.array(win, dtype=int)
        self.cum = np.concatenate([[0], np.cumsum(self.win)])

    def stats(self, cutoff_idx: int, window: int) -> tuple[int, int]:
        """(wins, n) over the last `window` signals whose contract had settled by candle `cutoff_idx`."""
        k = int(np.searchsorted(self.exp, cutoff_idx, side="right"))
        lo = max(0, k - window)
        return int(self.cum[k] - self.cum[lo]), k - lo


class Engine:
    def __init__(self, cfg: Config, signals: list[Signal]):
        self.cfg = cfg
        # With L2 on, the probability is estimated on the population L2 lets through (same class of trade).
        self.pop = Population(signals, cfg.spacing, only_l2=cfg.l2_enabled)

    def stage_a(self, sig: Signal, now_ts: int, risk: RiskManager) -> Decision | None:
        """Cheap checks needing no quote. Returns a SKIP Decision, or None to continue."""
        ok, why = risk.check(now_ts)
        if not ok:
            return Decision("SKIP", why, l2_pass=sig.l2_pass)
        if self.cfg.l2_enabled and not sig.l2_pass:
            import layer2_false_signal as l2
            return Decision("SKIP", l2.passes(sig.er, abs(sig.z), self.cfg)[1], l2_pass=False)
        return None

    def stage_b(self, sig: Signal, b: float) -> Decision:
        """Value, regime and sizing with the real (or assumed) payout ratio b."""
        cfg = self.cfg
        wins, n = self.pop.stats(sig.idx, cfg.estimation_window)
        d = Decision("SKIP", "", l2_pass=sig.l2_pass, payout_ratio=b)
        if cfg.l1_enabled:
            r = l1.evaluate(wins, n, b, cfg)
            d.n, d.p_hat, d.p_cons, d.breakeven = r.n, r.p_hat, r.p_cons, r.breakeven
            d.ep, d.el, d.ratio = r.ep, r.el, r.ratio
            if not r.passed:
                d.reason = r.reason
                return d
        else:
            d.n = n
            d.p_hat = wins / n if n else float("nan")
            d.p_cons = d.p_hat
            d.breakeven = 1.0 / (1.0 + b)
        ok, why = l4.passes(sig.atr_pct, cfg)
        if not ok:
            d.reason = why
            return d
        if cfg.l1_enabled:
            d.stake = l3.stake(d.p_cons, b, cfg)
        else:
            d.stake = round(min(cfg.stake_per_contract, cfg.max_stake), 2)
        d.action, d.reason = "TRADE", "all enabled layers passed"
        return d
