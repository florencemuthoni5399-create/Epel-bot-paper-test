"""Live runner. Two modes:
  signal : connects to Deriv's PUBLIC endpoint only. Pulls candles and real payout quotes, runs every layer and
           journals what it WOULD do. Places nothing. No token needed. START HERE.
  demo   : same, but buys contracts on your DEMO account (needs DERIV_TOKEN / DERIV_APP_ID / DERIV_ACCOUNT_ID).
There is no real-money mode in this project.

   python -m bot.live --mode signal
   python -m bot.live --mode demo
"""
from __future__ import annotations
import argparse
import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from config import Config
from data import load_csv, save_csv, merge
from deriv import WS, PUBLIC_WS, DerivError, fetch_candles, get_demo_ws_url, env
from engine import Engine, Decision
from indicators import Candles, zscore, efficiency_ratio
from journal import Journal
from risk import RiskManager
from strategy import find_signals, Signal

log = logging.getLogger("bot")
PAPER_ONLY = True  # hard lock in this distribution; do not enable order placement here


class Bot:
    def __init__(self, cfg: Config, mode: str, pub: WS, trade_ws: WS | None, journal: Journal, currency: str):
        self.cfg, self.mode, self.pub, self.trade_ws = cfg, mode, pub, trade_ws
        self.journal, self.currency = journal, currency
        self.risk = RiskManager(cfg)
        self._tasks: list[asyncio.Task] = []
        self.last_info: dict | None = None     # state of the most recent closed candle
        self.last_eval: dict | None = None     # most recent signal and what the layers decided

    # ---- payout quote (proposal needs no auth) ----
    async def quote(self, direction: str, stake: float, ws: WS | None = None) -> dict:
        ws = ws or self.pub
        r = await ws.request({"proposal": 1, "amount": stake, "basis": "stake", "contract_type": direction,
                              "currency": self.currency, "duration": self.cfg.contract_minutes, "duration_unit": "m",
                              "underlying_symbol": self.cfg.symbol})
        p = r["proposal"]
        return {"id": p["id"], "ask": float(p["ask_price"]), "b": float(p["payout"]) / float(p["ask_price"]) - 1.0}

    def restore_risk(self, now_ts: int) -> None:
        day_start = now_ts - now_ts % 86400
        pnl, trades, streak = self.journal.today(day_start)
        self.risk._roll(now_ts)
        self.risk.pnl_today, self.risk.trades_today, self.risk.consec_losses = pnl, trades, streak

    async def on_candles(self, candles: Candles, now_ts: int) -> Decision | None:
        """Called once per closed candle."""
        cfg = self.cfg
        resolved_paper = self.journal.resolve_shadow(candles, cfg)
        if self.mode == "signal":
            for settled_ts, paper_pnl in resolved_paper:
                self.risk.on_result(settled_ts, paper_pnl)
        last = len(candles) - 1
        z, er = zscore(candles.close, cfg.lookback)[-1], efficiency_ratio(candles.close, cfg.lookback)[-1]
        self.last_info = dict(epoch=int(candles.epoch[last]), close=float(candles.close[last]),
                              z=None if z != z else float(z), er=None if er != er else float(er),
                              band=cfg.num_std)
        sigs = find_signals(candles, cfg)
        if not sigs or sigs[-1].idx != last:
            log.info("candle %s closed: no signal", _t(candles.epoch[last]))
            return None
        sig = sigs[-1]
        if self.journal.seen(sig.epoch):
            return None
        engine = Engine(cfg, sigs)
        d = engine.stage_a(sig, now_ts, self.risk)
        if d is None:
            try:
                q = await self.quote(sig.direction, cfg.stake_per_contract)
                d = engine.stage_b(sig, q["b"])
            except (DerivError, KeyError, asyncio.TimeoutError, ConnectionError) as e:
                d = Decision("SKIP", f"quote unavailable: {e}", l2_pass=sig.l2_pass)
        log.info("SIGNAL %s z=%.2f -> %s | %s | n=%s p_cons=%.3f b/e=%.3f stake=%.2f", sig.direction, sig.z, d.action,
                 d.reason, d.n, d.p_cons, d.breakeven, d.stake)
        self.journal.log(sig, d, self.mode, now_ts)
        nn = lambda x: None if x is None or x != x else float(x)
        self.last_eval = dict(ts=now_ts, candle=sig.epoch, direction=sig.direction, z=nn(sig.z), action=d.action,
                              reason=d.reason, n=d.n, p_hat=nn(d.p_hat), p_cons=nn(d.p_cons), breakeven=nn(d.breakeven),
                              payout_ratio=nn(d.payout_ratio), ep=nn(d.ep), el=nn(d.el), ratio=nn(d.ratio), stake=d.stake)
        if d.action == "TRADE" and self.mode == "demo":
            await self.execute(sig, d, engine, now_ts)
        elif d.action == "TRADE":
            self.risk.on_open(now_ts, sig.epoch + cfg.granularity * (1 + cfg.hold_candles))  # paper: respect cooldown
        return d

    async def execute(self, sig: Signal, d: Decision, engine: Engine, now_ts: int) -> None:
        cfg = self.cfg
        try:
            q = await self.quote(sig.direction, d.stake, ws=self.trade_ws)
            if q["ask"] > d.stake * 1.02:
                log.warning("ask %.2f above stake %.2f - abort", q["ask"], d.stake); return
            d2 = engine.stage_b(sig, q["b"])      # re-check the gates against the quote we will actually pay
            if d2.action != "TRADE":
                log.warning("fresh quote failed gates (%s) - abort", d2.reason); return
            buy = await self.trade_ws.request({"buy": q["id"], "price": q["ask"]})
            cid = buy["buy"]["contract_id"]
        except (DerivError, KeyError, asyncio.TimeoutError, ConnectionError) as e:
            log.error("order failed: %s", e); return
        self.journal.set_contract(sig.epoch, cid)
        expiry = sig.epoch + cfg.granularity * (1 + cfg.hold_candles)
        self.risk.on_open(now_ts, expiry)
        log.info("BOUGHT demo contract %s %s stake %.2f", cid, sig.direction, d.stake)
        self._tasks.append(asyncio.create_task(self.track(sig.epoch, cid, expiry)))

    async def track(self, epoch: int, cid, expiry: int) -> None:
        try:
            q = await self.trade_ws.stream({"proposal_open_contract": 1, "contract_id": cid, "subscribe": 1})
            while True:
                msg = await asyncio.wait_for(q.get(), timeout=self.cfg.contract_minutes * 60 + 180)
                poc = msg.get("proposal_open_contract")
                if poc and poc.get("is_sold"):
                    pnl = float(poc["profit"])
                    self.journal.set_result(epoch, pnl > 0, pnl)
                    self.risk.on_result(int(time.time()), pnl)
                    log.info("RESULT contract %s pnl %.2f", cid, pnl)
                    return
        except Exception as e:
            log.error("tracking %s failed: %s (check the contract on Deriv; journal row left open)", cid, e)


def _t(epoch) -> str:
    return datetime.fromtimestamp(int(epoch), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


class Runner:
    """Owns the live loop so a web server (or the CLI) can start/stop it and read its status."""

    def __init__(self, cfg: Config, data_dir: str | None = None):
        self.cfg = cfg
        self.data_dir = data_dir or cfg.data_dir
        self.task: asyncio.Task | None = None
        self.bot: Bot | None = None
        self._sockets: list[WS] = []
        self.status: dict = dict(state="stopped", mode="off", connection="none", account=None, balance=None,
                                 currency=cfg.currency, last_cycle=None, last_error=None, candles=0, started=None)

    # ---- control ----
    async def start(self, mode: str) -> None:
        await self.stop()
        self.status.update(state="starting", mode=mode, last_error=None, started=int(time.time()))
        self.task = asyncio.create_task(self._supervise(mode))

    async def stop(self) -> None:
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except (asyncio.CancelledError, Exception):
                pass
            self.task = None
        await self._close()
        self.bot = None
        self.status.update(state="stopped", mode="off", connection="none")

    async def _close(self) -> None:
        for w in self._sockets:
            try:
                await w.close()
            except Exception:
                pass
        self._sockets = []

    # ---- loop ----
    async def _supervise(self, mode: str) -> None:
        while True:
            try:
                await self._session(mode)
                return
            except asyncio.CancelledError:
                raise
            except Exception as e:                       # reconnect after any failure, never spin
                log.error("session failed: %s", e)
                self.status.update(state="error", last_error=f"{type(e).__name__}: {e}")
                await self._close()
                await asyncio.sleep(30)

    async def _session(self, mode: str) -> None:
        if PAPER_ONLY and mode == "demo":
            raise RuntimeError("PAPER_ONLY is enabled; demo order placement is disabled in this build.")
        cfg = self.cfg
        os.makedirs(self.data_dir, exist_ok=True)
        pub = WS(PUBLIC_WS)
        await pub.open()
        self._sockets = [pub]
        trade_ws, currency = None, cfg.currency
        if mode == "demo":
            token, app_id, acct = os.environ.get("DERIV_TOKEN"), os.environ.get("DERIV_APP_ID"), os.environ.get("DERIV_ACCOUNT_ID")
            if not token or not app_id:
                self.status.update(state="error", last_error="Demo mode needs DERIV_TOKEN and DERIV_APP_ID environment variables.")
                raise asyncio.CancelledError()
            url, acct = await asyncio.to_thread(get_demo_ws_url, token, app_id, acct)
            trade_ws = WS(url)
            await trade_ws.open()
            self._sockets.append(trade_ws)
            self.status["account"] = acct
            try:
                bal = (await trade_ws.request({"balance": 1}))["balance"]
                currency = bal.get("currency", currency)
                self.status["balance"] = bal.get("balance")
            except Exception as e:
                log.warning("balance check failed (%s)", e)
        self.status.update(connection="demo" if mode == "demo" else "public", currency=currency)
        log.info("mode=%s symbol=%s", mode, cfg.symbol)

        path = f"{self.data_dir}/{cfg.symbol}_{cfg.granularity}.csv"
        try:
            candles = load_csv(path)
        except FileNotFoundError:
            candles = None
        self.status["state"] = "loading history"
        fresh = await fetch_candles(pub, cfg.symbol, cfg.granularity, cfg.history_candles if candles is None else 500,
                                    cfg.fetch_page_size)
        candles = merge(candles, fresh)
        self.bot = bot = Bot(cfg, mode, pub, trade_ws, Journal(f"{self.data_dir}/journal.db"), currency)
        bot.restore_risk(int(time.time()))
        g = cfg.granularity
        self.status.update(state="running", candles=len(candles))
        while True:
            await asyncio.sleep(g - time.time() % g + cfg.settle_seconds)
            try:
                new = await fetch_candles(pub, cfg.symbol, g, cfg.lookback + 60, cfg.fetch_page_size)
                candles = merge(candles, new)
                now = int(time.time())
                keep = candles.epoch + g <= now          # drop the still-forming candle
                closed = Candles(*(a[keep] for a in (candles.epoch, candles.open, candles.high, candles.low, candles.close)))
                save_csv(closed, path)
                self.status.update(last_cycle=now, candles=len(closed))
                if mode == "demo" and trade_ws:
                    try:
                        self.status["balance"] = (await trade_ws.request({"balance": 1}))["balance"].get("balance")
                    except Exception:
                        pass
                if now - int(closed.epoch[-1]) > 3 * g:
                    self.status["market"] = "closed/stale"
                    continue
                self.status["market"] = "open"
                await bot.on_candles(closed, now)
            except (DerivError, asyncio.TimeoutError) as e:
                log.error("cycle failed: %s (retry next candle)", e)
                self.status["last_error"] = str(e)
            except ConnectionError:
                raise                                     # let the supervisor reconnect


async def run(cfg: Config, mode: str) -> None:
    if PAPER_ONLY and mode == "demo":
        raise RuntimeError("PAPER_ONLY is enabled; demo order placement is disabled in this build.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(f"{cfg.data_dir}/bot.log")])
    if mode == "demo":
        env()  # fail fast with a clear message in CLI use
    await Runner(cfg)._session(mode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["signal", "demo"], default="signal")
    ap.add_argument("--config", default="config.json")
    a = ap.parse_args()
    cfg = Config.load(a.config)
    os.makedirs(cfg.data_dir, exist_ok=True)
    asyncio.run(run(cfg, a.mode))


if __name__ == "__main__":
    main()
