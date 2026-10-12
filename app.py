"""Web dashboard + API. Run locally:  uvicorn app:app --port 8000
Password-protected (HTTP Basic). Set DASHBOARD_PASSWORD. For local-only dev you can set DEV_NO_AUTH=1.
Demo only: this app has no real-money code path."""
from __future__ import annotations
import asyncio
import logging
import math
import os
import secrets
import sqlite3
import time
from contextlib import asynccontextmanager
from dataclasses import fields, replace
from pathlib import Path
from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from backtest import analyze, synthetic
from config import Config
from data import load_csv, save_csv
from deriv import WS, PUBLIC_WS, fetch_candles
from live import Runner
from stats import wilson_interval

log = logging.getLogger("bot")
DATA_DIR = os.environ.get("DATA_DIR", "data")
PAPER_ONLY = True  # hard lock: this package is paper-only, regardless of environment variables
CFG_PATH = f"{DATA_DIR}/config.json"
STATIC = Path(__file__).parent

# ---------------- config schema shown in the UI ----------------
SCHEMA = [
    ("Market & contract", [
        ("symbol", "Symbol", "Deriv symbol, e.g. frxEURUSD."),
        ("granularity", "Candle size (seconds)", "60, 120, 180, 300, 600, 900, 1800 or 3600. 300 = 5 minutes."),
        ("hold_candles", "Contract length (candles)", "Candles per contract. 3 candles x 5-minute candles = a 15-minute contract. Must make a whole number of minutes."),
    ]),
    ("SMC/ICT structural proxy (experimental)", [
        ("lookback", "Context lookback (candles)", "Used only for auxiliary z-score and efficiency-ratio features; SMC setup stages are fixed in the signal module."),
    ]),
    ("Layer 1: EP-EL value gate", [
        ("l1_enabled", "Enabled", "Turn off only to see the raw strategy."),
        ("min_samples", "Minimum settled signals", "No trade until this many past signals have resolved."),
        ("estimation_window", "Estimation window (signals)", "Use the most recent N resolved signals."),
        ("conf_z", "Confidence z", "1.645 = one-sided 95% lower bound (the 'conservative probability')."),
        ("min_prob_margin", "Minimum probability margin", "Conservative p must beat break-even by this much (0-0.5)."),
        ("max_loss_profit_ratio", "Max loss / profit ratio", "Expected loss divided by expected profit must be at most this."),
    ]),
    ("Layer 2: false-signal filter", [
        ("l2_enabled", "Enabled", "Disable if the Journal shows 'passed' does not beat 'rejected'."),
        ("er_max", "Max efficiency ratio", "Above this the market is trending; skip."),
        ("z_abs_max", "Max |z|", "Above this looks like a breakout spike; skip."),
    ]),
    ("Layer 3: fractional Kelly", [
        ("l3_enabled", "Enabled", "Off = flat floor stake."),
        ("kelly_fraction", "Kelly fraction", "0.25 = quarter Kelly."),
        ("risk_bankroll", "Sizing bankroll", "Bankroll used for sizing (not your demo balance)."),
        ("stake_per_contract", "Floor stake", "Minimum stake per trade."),
        ("max_stake", "Maximum stake", "Hard ceiling."),
    ]),
    ("Layer 4: volatility gate (off by default)", [
        ("l4_enabled", "Enabled", "Only enable if out-of-sample testing justifies it."),
        ("atr_pct_min", "Min ATR percentile", "Skip when volatility is below this percentile."),
        ("atr_pct_max", "Max ATR percentile", "Skip when volatility is above this percentile."),
    ]),
    ("Risk limits", [
        ("max_daily_loss", "Max daily loss", "Stop for the day at this loss (UTC day)."),
        ("max_trades_per_day", "Max trades per day", ""),
        ("max_consecutive_losses", "Max consecutive losses", "Pause after this many losses in a row."),
        ("loss_pause_minutes", "Pause length (minutes)", ""),
        ("cooldown_candles", "Cooldown (candles)", "No new trade for this many candles after one settles."),
    ]),
]
EDITABLE = {f for _, items in SCHEMA for f, _, _ in items}
GRANULARITIES = {60, 120, 180, 300, 600, 900, 1800, 3600}


def clean(o):
    """Make any structure JSON-safe (NaN/inf -> None)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


def J(o, status: int = 200) -> JSONResponse:
    return JSONResponse(clean(o), status_code=status)


def coerce(name: str, value, cfg: Config):
    cur = getattr(cfg, name)
    if isinstance(cur, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if isinstance(cur, int):
        v = float(value)
        if not math.isfinite(v) or v != int(v):
            raise ValueError(f"{name} must be a whole number")
        return int(v)
    if isinstance(cur, float):
        v = float(value)
        if not math.isfinite(v):
            raise ValueError(f"{name} must be a number")
        return v
    return str(value).strip()


def validate(cfg: Config) -> None:
    errs = []
    if cfg.granularity not in GRANULARITIES: errs.append("granularity must be one of " + ", ".join(map(str, sorted(GRANULARITIES))))
    if cfg.lookback < 5: errs.append("lookback must be at least 5")
    if cfg.num_std <= 0: errs.append("band width must be positive")
    if cfg.hold_candles < 1: errs.append("contract length must be at least 1 candle")
    elif (cfg.hold_candles * cfg.granularity) % 60 != 0 or cfg.hold_candles * cfg.granularity > 86400:
        errs.append("contract length must be a whole number of minutes and at most 24 hours")
    if not 0 <= cfg.min_prob_margin <= 0.5: errs.append("probability margin must be 0 to 0.5")
    if not 0 < cfg.kelly_fraction <= 1: errs.append("Kelly fraction must be above 0 and at most 1")
    if cfg.stake_per_contract <= 0 or cfg.max_stake < cfg.stake_per_contract: errs.append("stakes: need 0 < floor <= max")
    if cfg.max_stake > 100: errs.append("max stake above 100 is not allowed in this app")
    if cfg.max_loss_profit_ratio <= 0: errs.append("loss/profit ratio must be positive")
    if cfg.min_samples < 30: errs.append("minimum settled signals must be at least 30")
    if cfg.max_daily_loss <= 0 or cfg.max_trades_per_day < 1 or cfg.max_consecutive_losses < 1: errs.append("risk limits must be positive")
    if not cfg.symbol.replace("_", "").isalnum(): errs.append("symbol looks invalid")
    if errs:
        raise ValueError("; ".join(errs))


# ---------------- auth ----------------
security = HTTPBasic(auto_error=False)


def auth(creds: HTTPBasicCredentials | None = Depends(security)) -> None:
    pw = os.environ.get("DASHBOARD_PASSWORD")
    if not pw:
        if os.environ.get("DEV_NO_AUTH") == "1":
            return
        raise HTTPException(503, "Set the DASHBOARD_PASSWORD environment variable to use this dashboard.")
    if not creds or not secrets.compare_digest(creds.password.encode(), pw.encode()):
        raise HTTPException(401, "Login required", headers={"WWW-Authenticate": 'Basic realm="EP-EL bot"'})


# ---------------- app state ----------------
class State:
    cfg: Config
    runner: Runner
    bt_lock = asyncio.Lock()
    fetch = dict(state="idle", detail="", got=0)


S = State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    os.makedirs(DATA_DIR, exist_ok=True)
    S.cfg = Config.load(CFG_PATH)
    S.cfg.data_dir = DATA_DIR
    S.runner = Runner(S.cfg, DATA_DIR)
    mode = os.environ.get("BOT_AUTOSTART", "signal").lower()
    if PAPER_ONLY and mode == "demo":
        mode = "signal"  # never let a paper-only deployment start an order-capable mode
    if mode in ("signal", "demo"):
        await S.runner.start(mode)
    yield
    await S.runner.stop()


app = FastAPI(title="EP-EL bot", lifespan=lifespan)
api = APIRouter(dependencies=[Depends(auth)])


@app.get("/healthz")
async def healthz():
    return {"ok": True}


def db() -> sqlite3.Connection | None:
    p = f"{DATA_DIR}/journal.db"
    if not os.path.exists(p):
        return None
    c = sqlite3.connect(p)
    c.row_factory = sqlite3.Row
    return c


@api.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC / "index.html").read_text()


@api.get("/api/status")
async def status():
    r, bot = S.runner, S.runner.bot
    st = dict(r.status)
    st.update(now=int(time.time()), last_info=bot.last_info if bot else None, last_eval=bot.last_eval if bot else None,
              paper_only=PAPER_ONLY, strategy_label="SMC/ICT structural proxy (experimental)",
              demo_ready=(not PAPER_ONLY and bool(os.environ.get("DERIV_TOKEN") and os.environ.get("DERIV_APP_ID"))),
              symbol=S.cfg.symbol, granularity=S.cfg.granularity, contract_minutes=S.cfg.contract_minutes, lookback=S.cfg.lookback, num_std=S.cfg.num_std)
    if bot:
        k = bot.risk
        st["risk"] = dict(pnl_today=k.pnl_today, max_daily_loss=S.cfg.max_daily_loss, trades_today=k.trades_today,
                          max_trades=S.cfg.max_trades_per_day, streak=k.consec_losses,
                          max_streak=S.cfg.max_consecutive_losses, paused_until=k.paused_until, busy_until=k.busy_until)
    today = dict(pnl=0.0, wins=0, losses=0, trades=0, paper=0, screened=0, settled=0)
    c = db()
    if c:
        t0 = int(time.time()) - int(time.time()) % 86400
        for row in c.execute("SELECT action,mode,pnl,contract_win,shadow_win,stake,payout_ratio FROM decisions WHERE ts>=?", (t0,)):
            if row["action"] == "SKIP":
                today["screened"] += 1
            elif row["mode"] == "demo":
                today["trades"] += 1
                if row["contract_win"] is not None:
                    today["wins"] += row["contract_win"]; today["losses"] += 1 - row["contract_win"]
                    today["pnl"] += row["pnl"] or 0.0
            else:
                today["paper"] += 1
                if row["shadow_win"] is not None:
                    won = int(row["shadow_win"])
                    today["wins"] += won; today["losses"] += 1 - won; today["settled"] += 1
                    stake = float(row["stake"] or 0.0); payout = float(row["payout_ratio"] or 0.0)
                    today["pnl"] += stake * payout if won else -stake
        c.close()
    st["today"] = today
    return J(st)


@api.post("/api/control")
async def control(body: dict):
    mode = str(body.get("mode", "")).lower()
    if mode not in ("off", "signal", "demo"):
        raise HTTPException(400, "mode must be off, signal or demo")
    if PAPER_ONLY and mode == "demo":
        raise HTTPException(403, "This deployment is paper-only: demo order placement is disabled. Use Signal/Paper mode.")
    if mode == "demo" and not (os.environ.get("DERIV_TOKEN") and os.environ.get("DERIV_APP_ID")):
        raise HTTPException(400, "Demo trading needs DERIV_TOKEN and DERIV_APP_ID set as environment variables.")
    if mode == "off":
        await S.runner.stop()
    else:
        await S.runner.start(mode)
    return J({"ok": True, "mode": mode})


@api.get("/api/journal")
async def journal(limit: int = 100):
    c = db()
    out = dict(rows=[], l2={}, calibration=[], summary={})
    if not c:
        return J(out)
    limit = max(1, min(limit, 500))
    rows = [dict(r) for r in c.execute("SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,))]
    for r in rows:
        r["result"] = r["contract_win"] if r["contract_win"] is not None else r["shadow_win"]
        r["result_source"] = "contract" if r["contract_win"] is not None else ("candles" if r["shadow_win"] is not None else None)
    out["rows"] = rows
    allr = [dict(r) for r in c.execute("SELECT * FROM decisions")]
    out["summary"] = dict(total=len(allr), taken=sum(r["action"] == "TRADE" for r in allr))
    for name, flag in (("passed", 1), ("rejected", 0)):
        g = [r for r in allr if r["l2_pass"] == flag and r["shadow_win"] is not None]
        if g:
            w = sum(r["shadow_win"] for r in g); lo, hi = wilson_interval(w, len(g))
            out["l2"][name] = dict(n=len(g), rate=w / len(g), lo=lo, hi=hi)
    done = [r for r in allr if r["action"] == "TRADE" and (r["contract_win"] is not None or r["shadow_win"] is not None)]
    for a, z in zip([0, .55, .6, .65, .7], [.55, .6, .65, .7, 1.01]):
        g = [r for r in done if r["p_cons"] is not None and a <= r["p_cons"] < z]
        if g:
            out["calibration"].append(dict(lo=a, hi=min(z, 1.0), n=len(g),
                                           realised=sum((r["contract_win"] if r["contract_win"] is not None else r["shadow_win"]) for r in g) / len(g)))
    c.close()
    return J(out)


@api.get("/api/config")
async def get_config():
    vals = {f.name: getattr(S.cfg, f.name) for f in fields(Config) if f.name in EDITABLE}
    return J(dict(values=vals, schema=[dict(title=t, fields=[dict(name=n, label=l, help=h, kind=type(getattr(S.cfg, n)).__name__) for n, l, h in items]) for t, items in SCHEMA]))


@api.post("/api/config")
async def set_config(body: dict):
    new = replace(S.cfg)
    try:
        for k, v in body.items():
            if k not in EDITABLE:
                raise ValueError(f"'{k}' cannot be changed here")
            setattr(new, k, coerce(k, v, S.cfg))
        validate(new)
    except (ValueError, TypeError) as e:
        raise HTTPException(400, str(e))
    restart = (new.symbol, new.granularity) != (S.cfg.symbol, S.cfg.granularity)
    for f in fields(Config):          # mutate in place: the running bot holds this same object
        if f.name in EDITABLE:
            setattr(S.cfg, f.name, getattr(new, f.name))
    S.cfg.dump(CFG_PATH)
    mode = S.runner.status.get("mode", "off")
    if restart and mode in ("signal", "demo"):
        await S.runner.start(mode)
    return J({"ok": True, "restarted": restart and mode != "off"})


@api.get("/api/quote")
async def live_quote():
    """Real payout Deriv quotes right now for the configured contract (public endpoint, nothing is bought)."""
    ws, out = WS(PUBLIC_WS), {}
    try:
        await ws.open()
        for side in ("CALL", "PUT"):
            r = await ws.request({"proposal": 1, "amount": 1, "basis": "stake", "contract_type": side,
                                  "currency": S.cfg.currency, "duration": S.cfg.contract_minutes, "duration_unit": "m",
                                  "underlying_symbol": S.cfg.symbol}, timeout=15)
            p = r["proposal"]
            b = float(p["payout"]) / float(p["ask_price"]) - 1.0
            out[side] = dict(b=b, breakeven=1.0 / (1.0 + b))
    except Exception as e:
        raise HTTPException(502, f"Could not get a quote ({type(e).__name__}: {e}). The market may be closed.")
    finally:
        await ws.close()
    out["minutes"] = S.cfg.contract_minutes
    out["symbol"] = S.cfg.symbol
    return J(out)


@api.post("/api/backtest")
async def backtest(body: dict):
    src = str(body.get("source", "stored"))
    try:
        b = float(body.get("payout", 0.85))
    except (TypeError, ValueError):
        raise HTTPException(400, "payout must be a number")
    if not 0.3 <= b <= 1.5:
        raise HTTPException(400, "payout must be between 0.3 and 1.5 (net profit per $1 staked)")
    if S.bt_lock.locked():
        raise HTTPException(429, "A backtest is already running")
    async with S.bt_lock:
        if src.startswith("synthetic_"):
            kind = src.split("_", 1)[1]
            if kind not in ("random", "meanrev", "strong"):
                raise HTTPException(400, "unknown synthetic source")
            c = synthetic(kind)
        else:
            path = f"{DATA_DIR}/{S.cfg.symbol}_{S.cfg.granularity}.csv"
            if not os.path.exists(path):
                raise HTTPException(404, "No stored candles yet. Use 'Download history' first.")
            c = load_csv(path)
        if len(c) < S.cfg.lookback * 20:
            raise HTTPException(400, "Not enough candles to backtest")
        cfg = replace(S.cfg)
        res = await asyncio.to_thread(analyze, c, cfg, b, bool(body.get("compare_l4")))
    res["source"] = src
    return J(res)


async def _fetch_job(n: int) -> None:
    S.fetch.update(state="running", detail="connecting", got=0)
    ws = WS(PUBLIC_WS)
    try:
        await ws.open()
        S.fetch["detail"] = "downloading"
        c = await fetch_candles(ws, S.cfg.symbol, S.cfg.granularity, n, S.cfg.fetch_page_size)
        save_csv(c, f"{DATA_DIR}/{S.cfg.symbol}_{S.cfg.granularity}.csv")
        S.fetch.update(state="done", detail=f"saved {len(c)} candles", got=len(c))
    except Exception as e:
        S.fetch.update(state="error", detail=f"{type(e).__name__}: {e}")
    finally:
        await ws.close()


@api.post("/api/fetch")
async def start_fetch(body: dict):
    if S.fetch["state"] == "running":
        raise HTTPException(429, "Download already running")
    n = max(1000, min(int(body.get("candles", 30000)), 100000))
    asyncio.create_task(_fetch_job(n))
    return J({"ok": True})


@api.get("/api/fetch")
async def fetch_status():
    path = f"{DATA_DIR}/{S.cfg.symbol}_{S.cfg.granularity}.csv"
    stored = None
    if os.path.exists(path):
        stored = sum(1 for _ in open(path)) - 1
    return J(dict(S.fetch, stored=stored))


app.include_router(api)
