"""LAYER 3 - fractional Kelly staking (money management only; it creates no edge).
For a binary payout with win prob p and net payout b:  f* = (p*b - (1-p)) / b.
We use the CONSERVATIVE p from Layer 1, a fraction of f*, a fixed sizing bankroll,
a floor, and a hard ceiling. Kelly never rescues a trade that Layer 1 rejected."""
from __future__ import annotations
from config import Config


def full_kelly(p: float, b: float) -> float:
    return max(0.0, (p * b - (1.0 - p)) / b)


def stake(p_cons: float, b: float, cfg: Config) -> float:
    if not cfg.l3_enabled:
        return round(min(cfg.stake_per_contract, cfg.max_stake), 2)
    raw = cfg.risk_bankroll * cfg.kelly_fraction * full_kelly(p_cons, b)
    return round(min(max(raw, cfg.stake_per_contract), cfg.max_stake), 2)
