"""Minimal Deriv client for the new REST + OTP-WebSocket API (developers.deriv.com).

SAFETY: this module can only produce a DEMO trading socket. get_demo_ws_url() refuses any URL that is not
the /ws/demo endpoint, and there is no code path anywhere in this project that opens /ws/real.

Env vars (never hard-code or commit tokens):
  DERIV_TOKEN       Personal Access Token (needs the 'trade' scope)
  DERIV_APP_ID      your Deriv app id (required for PAT auth)
  DERIV_ACCOUNT_ID  your DEMO options account id (run `python -m bot.deriv --list-accounts` to see them)
"""
from __future__ import annotations
import asyncio
import json
import os
import urllib.error
import urllib.request
import websockets
from data import from_rows
from indicators import Candles

REST_BASE = "https://api.derivws.com"
PUBLIC_WS = "wss://api.derivws.com/trading/v1/options/ws/public"


class DerivError(Exception):
    pass


class WS:
    """One shared socket; replies are matched to requests by req_id."""

    def __init__(self, url: str):
        self.url, self.ws = url, None
        self._id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._streams: dict[int, asyncio.Queue] = {}
        self._tasks: list[asyncio.Task] = []

    async def open(self) -> None:
        self.ws = await websockets.connect(self.url, max_size=None)
        self._tasks = [asyncio.create_task(self._reader()), asyncio.create_task(self._pinger())]

    async def close(self) -> None:
        for t in self._tasks:
            t.cancel()
        if self.ws:
            await self.ws.close()

    async def _reader(self) -> None:
        try:
            async for raw in self.ws:
                msg = json.loads(raw)
                rid = msg.get("req_id")
                if rid in self._streams:
                    self._streams[rid].put_nowait(msg)
                    continue
                fut = self._pending.pop(rid, None)
                if fut and not fut.done():
                    if "error" in msg:
                        e = msg["error"]
                        fut.set_exception(DerivError(f"{e.get('code')}: {e.get('message')}"))
                    else:
                        fut.set_result(msg)
        except Exception as exc:  # connection dropped
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionError(f"socket closed: {exc}"))
            self._pending.clear()

    async def _pinger(self) -> None:
        while True:
            await asyncio.sleep(25)
            await self.ws.send(json.dumps({"ping": 1}))

    async def request(self, payload: dict, timeout: float = 20) -> dict:
        self._id += 1
        rid = self._id
        fut = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        await self.ws.send(json.dumps({**payload, "req_id": rid}))
        try:
            return await asyncio.wait_for(fut, timeout)
        finally:
            self._pending.pop(rid, None)

    async def stream(self, payload: dict) -> asyncio.Queue:
        self._id += 1
        rid = self._id
        q: asyncio.Queue = asyncio.Queue()
        self._streams[rid] = q
        await self.ws.send(json.dumps({**payload, "req_id": rid}))
        return q


async def fetch_candles(ws: WS, symbol: str, granularity: int, count: int, page: int = 1000) -> Candles:
    """Page backwards through ticks_history (style=candles). Candle `epoch` is the candle OPEN time."""
    rows: dict[int, dict] = {}
    end = "latest"
    while len(rows) < count:
        resp = await ws.request({"ticks_history": symbol, "style": "candles", "granularity": granularity,
                                 "count": min(page, count - len(rows)), "end": end})
        got = resp.get("candles") or []
        if not got:
            break
        before = len(rows)
        for r in got:
            rows[int(r["epoch"])] = r
        if len(rows) == before:
            break
        end = str(min(int(r["epoch"]) for r in got) - 1)
    return from_rows(list(rows.values()))


# ---------------- REST (account lookup + OTP) ----------------
def rest(method: str, path: str, token: str, app_id: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        REST_BASE + path, method=method, data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {token}", "Deriv-App-ID": app_id, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise DerivError(f"HTTP {e.code} on {path}: {e.read()[:300]!r}")


def _walk(o):
    if isinstance(o, dict):
        yield o
        for v in o.values():
            yield from _walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk(v)


def _find_wss(o) -> str | None:
    if isinstance(o, str) and o.startswith("wss://"):
        return o
    if isinstance(o, dict):
        for v in o.values():
            r = _find_wss(v)
            if r:
                return r
    if isinstance(o, list):
        for v in o:
            r = _find_wss(v)
            if r:
                return r
    return None


def guess_demo_account(accounts_json: dict) -> str | None:
    """The response schema isn't pinned here, so look for an account dict that looks like a demo account."""
    for d in _walk(accounts_json):
        ident = next((str(d[k]) for k in ("account_id", "accountId", "id", "loginid") if k in d), None)
        if not ident:
            continue
        vals = [str(v).lower() for v in d.values() if isinstance(v, (str, int, bool))]
        if "demo" in vals or any(d.get(k) for k in ("is_virtual", "is_demo", "demo")) or ident.upper().startswith(("DOT", "VRT")):
            return ident
    return None


def get_demo_ws_url(token: str, app_id: str, account_id: str | None = None) -> tuple[str, str]:
    if not account_id:
        account_id = guess_demo_account(rest("GET", "/trading/v1/options/accounts", token, app_id))
        if not account_id:
            raise DerivError("Could not find a demo account. Set DERIV_ACCOUNT_ID (see --list-accounts).")
    otp = rest("POST", f"/trading/v1/options/accounts/{account_id}/otp", token, app_id, {})
    url = _find_wss(otp)
    if not url:
        raise DerivError("OTP response had no wss:// URL.")
    if "/ws/demo" not in url or "/ws/real" in url:
        raise DerivError("REFUSING: OTP URL is not a demo endpoint. This bot is demo-only.")
    return url, account_id


def env() -> tuple[str, str, str | None]:
    token, app_id = os.environ.get("DERIV_TOKEN"), os.environ.get("DERIV_APP_ID")
    if not token or not app_id:
        raise SystemExit("Set DERIV_TOKEN and DERIV_APP_ID in your environment (never in code).")
    return token, app_id, os.environ.get("DERIV_ACCOUNT_ID")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-accounts", action="store_true")
    a = ap.parse_args()
    if a.list_accounts:
        t, app, _ = env()
        print(json.dumps(rest("GET", "/trading/v1/options/accounts", t, app), indent=2))
