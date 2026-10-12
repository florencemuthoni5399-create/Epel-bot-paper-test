"""SQLite journal. Every signal is logged - taken or skipped - with the reason. Skipped signals also get a
'shadow' outcome computed from candles, so you can later check whether your filters really removed losers."""
from __future__ import annotations
import sqlite3
from pathlib import Path
import numpy as np
from config import Config
from indicators import Candles

SCHEMA = """CREATE TABLE IF NOT EXISTS decisions(
 id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, candle_epoch INTEGER UNIQUE, direction TEXT, z REAL, er REAL,
 atr_pct REAL, l2_pass INTEGER, action TEXT, mode TEXT, reason TEXT, n INTEGER, p_hat REAL, p_cons REAL,
 breakeven REAL, payout_ratio REAL, ep REAL, el REAL, ratio REAL, stake REAL, contract_id TEXT,
 pnl REAL, contract_win INTEGER, shadow_win INTEGER)"""


def _nn(x):
    return None if x is None or x != x else float(x)


class Journal:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute(SCHEMA)
        self.db.commit()

    def seen(self, candle_epoch: int) -> bool:
        return self.db.execute("SELECT 1 FROM decisions WHERE candle_epoch=?", (candle_epoch,)).fetchone() is not None

    def log(self, sig, d, mode: str, ts: int) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO decisions(ts,candle_epoch,direction,z,er,atr_pct,l2_pass,action,mode,reason,n,p_hat,p_cons,"
            "breakeven,payout_ratio,ep,el,ratio,stake) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (ts, sig.epoch, sig.direction, _nn(sig.z), _nn(sig.er), _nn(sig.atr_pct), int(d.l2_pass), d.action, mode, d.reason,
             d.n, _nn(d.p_hat), _nn(d.p_cons), _nn(d.breakeven), _nn(d.payout_ratio), _nn(d.ep), _nn(d.el), _nn(d.ratio), d.stake))
        self.db.commit()

    def set_contract(self, candle_epoch: int, contract_id: str) -> None:
        self.db.execute("UPDATE decisions SET contract_id=? WHERE candle_epoch=?", (str(contract_id), candle_epoch))
        self.db.commit()

    def set_result(self, candle_epoch: int, win: bool, pnl: float) -> None:
        self.db.execute("UPDATE decisions SET contract_win=?, pnl=? WHERE candle_epoch=?", (int(win), pnl, candle_epoch))
        self.db.commit()

    def resolve_shadow(self, c: Candles, cfg: Config) -> int:
        rows = self.db.execute("SELECT candle_epoch,direction FROM decisions WHERE shadow_win IS NULL").fetchall()
        done = 0
        for epoch, direction in rows:
            i = int(np.searchsorted(c.epoch, epoch))
            if i >= len(c) or c.epoch[i] != epoch:
                continue
            exp = i + cfg.hold_candles
            if exp >= len(c):
                continue
            entry, exit_ = c.open[i + 1], c.close[exp]
            win = exit_ < entry if direction == "PUT" else exit_ > entry
            self.db.execute("UPDATE decisions SET shadow_win=? WHERE candle_epoch=?", (int(win), epoch))
            done += 1
        self.db.commit()
        return done

    def today(self, day_start_ts: int) -> tuple[float, int, int]:
        """(estimated/realized pnl, opened trades, consecutive resolved losses) for today's entries."""
        rows = self.db.execute(
            "SELECT mode,pnl,contract_win,shadow_win,stake,payout_ratio FROM decisions "
            "WHERE action='TRADE' AND ts>=? ORDER BY id", (day_start_ts,)
        ).fetchall()
        pnl, resolved = 0.0, []
        for mode, actual_pnl, contract_win, shadow_win, stake, payout_ratio in rows:
            if mode == "demo" and contract_win is not None:
                pnl += float(actual_pnl or 0.0)
                resolved.append(float(actual_pnl or 0.0))
            elif mode == "signal" and shadow_win is not None:
                won = bool(shadow_win)
                paper_pnl = float(stake or 0.0) * float(payout_ratio or 0.0) if won else -float(stake or 0.0)
                pnl += paper_pnl
                resolved.append(paper_pnl)
        streak = 0
        for value in reversed(resolved):
            if value < 0:
                streak += 1
            else:
                break
        return pnl, len(rows), streak

    def resolve_shadow(self, c: Candles, cfg: Config) -> list[tuple[int, float]]:
        """Resolve pending candle outcomes and return (settlement timestamp, P&L) for taken paper entries."""
        rows = self.db.execute(
            "SELECT candle_epoch,direction,action,mode,stake,payout_ratio FROM decisions WHERE shadow_win IS NULL"
        ).fetchall()
        outcomes: list[tuple[int, float]] = []
        for epoch, direction, action, mode, stake, payout_ratio in rows:
            i = int(np.searchsorted(c.epoch, epoch))
            if i >= len(c) or c.epoch[i] != epoch:
                continue
            exp = i + cfg.hold_candles
            if exp >= len(c):
                continue
            entry, exit_ = c.open[i + 1], c.close[exp]
            win = exit_ < entry if direction == "PUT" else exit_ > entry
            self.db.execute("UPDATE decisions SET shadow_win=? WHERE candle_epoch=?", (int(win), epoch))
            if action == "TRADE" and mode == "signal":
                stake_value = float(stake or 0.0)
                b = float(payout_ratio or 0.0)
                pnl = stake_value * b if win else -stake_value
                outcomes.append((int(c.epoch[exp]) + cfg.granularity, pnl))
        self.db.commit()
        return outcomes
