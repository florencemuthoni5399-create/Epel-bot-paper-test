"""LAYER 1 - EP-EL value filter (risk-and-value gate).
Per $1 staked, with conservative win probability p and payout ratio b:
    EP = p * b          (expected profit)
    EL = (1 - p)        (expected loss)
    EP - EL = p(1+b) - 1
So the gate is really: is the *conservative* p above break-even 1/(1+b) by a margin,
and is EL/EP small enough. p is a Wilson lower bound, never the raw hit rate."""
from __future__ import annotations
from dataclasses import dataclass
from config import Config
from stats import wilson_lower, breakeven


@dataclass
class L1Result:
    n: int
    p_hat: float
    p_cons: float
    breakeven: float
    ep: float
    el: float
    ratio: float
    passed: bool
    reason: str


def evaluate(wins: int, n: int, b: float, cfg: Config) -> L1Result:
    be = breakeven(b)
    p_hat = wins / n if n else float("nan")
    p_cons = wilson_lower(wins, n, cfg.conf_z) if n else 0.0
    ep, el = p_cons * b, 1.0 - p_cons
    ratio = el / ep if ep > 0 else float("inf")
    if n < cfg.min_samples:
        return L1Result(n, p_hat, p_cons, be, ep, el, ratio, False,
                        f"L1: only {n} resolved signals (need {cfg.min_samples})")
    if p_cons < be + cfg.min_prob_margin:
        return L1Result(n, p_hat, p_cons, be, ep, el, ratio, False,
                        f"L1: conservative p {p_cons:.3f} < break-even {be:.3f} + margin {cfg.min_prob_margin:.2f}")
    if ratio > cfg.max_loss_profit_ratio:
        return L1Result(n, p_hat, p_cons, be, ep, el, ratio, False,
                        f"L1: EL/EP {ratio:.2f} > max {cfg.max_loss_profit_ratio:.2f}")
    return L1Result(n, p_hat, p_cons, be, ep, el, ratio, True, "L1: passes EP-EL gate")
