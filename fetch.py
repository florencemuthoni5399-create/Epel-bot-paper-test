"""Download historical candles from Deriv's PUBLIC endpoint (no login needed) to CSV for backtesting.
   python -m bot.fetch --candles 30000"""
from __future__ import annotations
import argparse
import asyncio
from config import Config
from data import save_csv
from deriv import WS, PUBLIC_WS, fetch_candles


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json"); ap.add_argument("--candles", type=int)
    a = ap.parse_args()
    cfg = Config.load(a.config)
    n = a.candles or cfg.history_candles
    ws = WS(PUBLIC_WS)
    await ws.open()
    try:
        c = await fetch_candles(ws, cfg.symbol, cfg.granularity, n, cfg.fetch_page_size)
    finally:
        await ws.close()
    path = f"{cfg.data_dir}/{cfg.symbol}_{cfg.granularity}.csv"
    save_csv(c, path)
    print(f"Saved {len(c)} candles to {path}")


if __name__ == "__main__":
    asyncio.run(main())
