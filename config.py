"""All tunable settings live here. Thresholds are meant to be set BEFORE looking at
backtest results (pre-registered) and then left alone, to avoid overfitting."""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict, fields
from pathlib import Path


@dataclass
class Config:
    # ---- market / contract ----
    symbol: str = "frxEURUSD"
    granularity: int = 300          # candle size in seconds (300 = 5 min)
    hold_candles: int = 3           # contract lasts this many candles (3 x 5 min = a 15-minute contract)
    currency: str = "USD"

    # ---- primary strategy: Bollinger mean reversion ----
    lookback: int = 40
    num_std: float = 2.0
    fresh_cross: bool = True        # only the first candle that closes outside the band (independence)

    # ---- Layer 1: EP-EL value filter ----
    l1_enabled: bool = True
    min_samples: int = 200          # resolved signals needed before any estimate is trusted
    estimation_window: int = 1000   # use the most recent N resolved signals
    conf_z: float = 1.645           # Wilson lower bound (one-sided ~95%) = "conservative probability"
    min_prob_margin: float = 0.05   # conservative p must beat break-even by this much
    max_loss_profit_ratio: float = 0.8   # expected loss / expected profit must be <= this

    # ---- Layer 2: false-signal filter (trend / breakout detector) ----
    l2_enabled: bool = True
    er_max: float = 0.35            # Kaufman efficiency ratio over lookback; above = trending, skip
    z_abs_max: float = 3.5          # |z| above this looks like a breakout/news spike, skip

    # ---- Layer 3: fractional Kelly ----
    l3_enabled: bool = True
    kelly_fraction: float = 0.25
    risk_bankroll: float = 100.0    # bankroll used for sizing (NOT the demo balance)
    stake_per_contract: float = 1.0 # floor stake (and flat stake when Kelly is off)
    max_stake: float = 5.0          # hard ceiling

    # ---- Layer 4: volatility / regime gate (OFF until out-of-sample testing justifies it) ----
    l4_enabled: bool = False
    atr_period: int = 14
    regime_lookback: int = 500
    atr_pct_min: float = 10.0
    atr_pct_max: float = 90.0

    # ---- risk limits ----
    max_daily_loss: float = 10.0
    max_trades_per_day: int = 15
    max_consecutive_losses: int = 3
    loss_pause_minutes: int = 120
    cooldown_candles: int = 2       # no new trade for this many candles after one opens

    # ---- live runner ----
    history_candles: int = 30000
    fetch_page_size: int = 1000
    settle_seconds: int = 3
    data_dir: str = "data"

    @property
    def contract_minutes(self) -> int:
        return max(1, self.hold_candles * self.granularity // 60)

    @property
    def spacing(self) -> int:
        """Minimum candle gap between two trades (hold + cooldown)."""
        return self.hold_candles + self.cooldown_candles

    @classmethod
    def load(cls, path: str | None) -> "Config":
        cfg = cls()
        if path and Path(path).exists():
            raw = json.loads(Path(path).read_text())
            names = {f.name for f in fields(cls)}
            unknown = set(raw) - names
            if unknown:
                raise ValueError(f"Unknown config keys: {sorted(unknown)}")
            for k, v in raw.items():
                setattr(cfg, k, v)
        return cfg

    def dump(self, path: str) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))
