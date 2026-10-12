"""Hard risk limits. Time is passed in explicitly so the backtest and the live bot share this code."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from config import Config


def _day(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


@dataclass
class RiskManager:
    cfg: Config
    day: str = ""
    pnl_today: float = 0.0
    trades_today: int = 0
    consec_losses: int = 0
    paused_until: int = 0
    busy_until: int = 0      # no new trade until this timestamp (open position + cooldown)

    def _roll(self, ts: int) -> None:
        d = _day(ts)
        if d != self.day:
            self.day, self.pnl_today, self.trades_today = d, 0.0, 0
            self.consec_losses = 0   # a new day starts clean

    def check(self, ts: int) -> tuple[bool, str]:
        self._roll(ts)
        if ts < self.busy_until:
            return False, "RISK: position open or cooling down"
        if ts < self.paused_until:
            return False, "RISK: paused after consecutive losses"
        if self.pnl_today <= -self.cfg.max_daily_loss:
            return False, f"RISK: daily loss limit reached ({self.pnl_today:.2f})"
        if self.trades_today >= self.cfg.max_trades_per_day:
            return False, "RISK: max trades per day reached"
        return True, "RISK: ok"

    def on_open(self, ts: int, expiry_ts: int) -> None:
        self._roll(ts)
        self.trades_today += 1
        self.busy_until = expiry_ts + self.cfg.cooldown_candles * self.cfg.granularity

    def on_result(self, ts: int, pnl: float) -> None:
        self._roll(ts)
        self.pnl_today += pnl
        if pnl < 0:
            self.consec_losses += 1
            if self.consec_losses >= self.cfg.max_consecutive_losses:
                self.paused_until = ts + self.cfg.loss_pause_minutes * 60
                self.consec_losses = 0
        else:
            self.consec_losses = 0
