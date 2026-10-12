"""Read the journal: how often it trades, why it skips, and whether predicted probabilities match reality.
   python -m bot.report"""
from __future__ import annotations
import argparse
import sqlite3
from collections import Counter
from config import Config
from stats import wilson_interval


def outcome(r):  # contract result if we have it, else candle-based shadow result
    cw, sw = r["contract_win"], r["shadow_win"]
    return cw if cw is not None else sw


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.json")
    cfg = Config.load(ap.parse_args().config)
    db = sqlite3.connect(f"{cfg.data_dir}/journal.db"); db.row_factory = sqlite3.Row
    rows = db.execute("SELECT * FROM decisions").fetchall()
    if not rows:
        print("Journal is empty."); return
    trades = [r for r in rows if r["action"] == "TRADE"]
    print(f"Signals logged: {len(rows)}   taken: {len(trades)}   skipped: {len(rows)-len(trades)}")
    print("\nWhy signals were skipped:")
    for k, v in Counter(r["reason"].split(" (")[0].split(":")[0] + ":" + r["reason"].split(":", 1)[-1][:40] for r in rows if r["action"] == "SKIP").most_common(8):
        print(f"  {v:>4}  {k}")
    done = [r for r in trades if outcome(r) is not None]
    if done:
        w = sum(outcome(r) for r in done); lo, hi = wilson_interval(w, len(done))
        pnl = sum(r["pnl"] for r in done if r["pnl"] is not None)
        print(f"\nTaken & resolved: {len(done)}  win {100*w/len(done):.1f}%  CI95 [{lo*100:.1f},{hi*100:.1f}]  demo P/L {pnl:.2f}")
        print("Calibration (conservative predicted p vs realised):")
        for a, z in zip([0, .55, .6, .65, .7], [.55, .6, .65, .7, 1.01]):
            g = [r for r in done if r["p_cons"] is not None and a <= r["p_cons"] < z]
            if g:
                print(f"  {a:.2f}-{min(z,1):.2f}: n={len(g):<4} realised {100*sum(outcome(r) for r in g)/len(g):.1f}%")
        beavg = sum(r["breakeven"] for r in done if r["breakeven"]) / max(1, len([r for r in done if r["breakeven"]]))
        print(f"Average break-even of taken trades: {beavg*100:.1f}%  (realised win rate must beat this)")
    print("\nDoes Layer 2 remove losers? (candle-based outcomes of every logged signal)")
    for name, flag in (("L2 passed", 1), ("L2 rejected", 0)):
        g = [r for r in rows if r["l2_pass"] == flag and r["shadow_win"] is not None]
        if g:
            w = sum(r["shadow_win"] for r in g); lo, hi = wilson_interval(w, len(g))
            print(f"  {name:<12} n={len(g):<4} hit {100*w/len(g):.1f}%  CI95 [{lo*100:.1f},{hi*100:.1f}]")


if __name__ == "__main__":
    main()
