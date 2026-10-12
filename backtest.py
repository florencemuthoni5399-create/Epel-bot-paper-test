"""Walk-forward backtest of the layered bot.

Honest-by-construction rules:
  * entry = open of the candle AFTER the signal (no trading the close you just saw)
  * a tie loses
  * the probability used for each decision comes only from signals already settled at that moment
  * one position at a time, cooldown, daily limits are enforced by the same RiskManager as live
  * payout is an ASSUMPTION here (live uses the real quote) - so a payout sensitivity table is printed

Usage:
  python -m bot.backtest --csv data/frxEURUSD_300.csv --payout 0.85
  python -m bot.backtest --synthetic random      # sanity check: bot should find nothing
  python -m bot.backtest --synthetic meanrev     # weak planted edge: gates should mostly decline
  python -m bot.backtest --synthetic strong      # strong planted edge: layers should trade
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, replace
import numpy as np
from config import Config
from indicators import Candles
from strategy import find_signals, Signal
from engine import Engine, Population
from risk import RiskManager
from stats import wilson_interval, breakeven
from data import load_csv
import layer1_epel


@dataclass
class Trade:
    idx: int
    epoch: int
    direction: str
    stake: float
    win: bool
    pnl: float
    p_cons: float


def simulate(c: Candles, cfg: Config, signals: list[Signal], b: float) -> list[Trade]:
    engine, risk = Engine(cfg, signals), RiskManager(cfg)
    out: list[Trade] = []
    for s in signals:
        if s.win is None:
            continue
        t_dec = s.epoch + cfg.granularity
        d = engine.stage_a(s, t_dec, risk)
        if d is None:
            d = engine.stage_b(s, b)
        if d.action != "TRADE":
            continue
        pnl = d.stake * b if s.win else -d.stake
        expiry_ts = int(c.epoch[s.exp_idx]) + cfg.granularity
        risk.on_open(t_dec, expiry_ts)
        risk.on_result(expiry_ts, pnl)
        out.append(Trade(s.idx, s.epoch, s.direction, d.stake, s.win, pnl, d.p_cons))
    return out


def summarize(trades: list[Trade], b: float) -> dict:
    n = len(trades)
    if n == 0:
        return dict(n=0)
    wins = sum(t.win for t in trades)
    lo, hi = wilson_interval(wins, n)
    pnl = np.cumsum([t.pnl for t in trades])
    staked = sum(t.stake for t in trades)
    dd = float(np.max(np.maximum.accumulate(pnl) - pnl)) if n else 0.0
    return dict(n=n, wins=wins, wr=wins / n, lo=lo, hi=hi, be=breakeven(b), pnl=float(pnl[-1]),
                roi=float(pnl[-1]) / staked if staked else 0.0, dd=dd)


def fmt(label: str, s: dict) -> str:
    if s["n"] == 0:
        return f"  {label:<34} no trades"
    return (f"  {label:<34} n={s['n']:<5} win={s['wr']*100:5.1f}%  CI95=[{s['lo']*100:4.1f},{s['hi']*100:4.1f}]  "
            f"B/E={s['be']*100:4.1f}%  P/L={s['pnl']:8.2f}  ROI={s['roi']*100:6.1f}%  maxDD={s['dd']:6.2f}")


def variants(cfg: Config, compare_l4: bool) -> list[tuple[str, Config]]:
    base = replace(cfg, l1_enabled=False, l2_enabled=False, l3_enabled=False, l4_enabled=False)
    v = [("0. Raw strategy (no layers)", base),
         ("1. + L1 EP-EL gate", replace(base, l1_enabled=True)),
         ("2. + L2 false-signal filter", replace(base, l1_enabled=True, l2_enabled=True)),
         ("3. + L3 fractional Kelly", replace(base, l1_enabled=True, l2_enabled=True, l3_enabled=True))]
    if compare_l4:
        v.append(("4. + L4 regime gate", replace(v[-1][1], l4_enabled=True)))
    return v


def analyze(c: Candles, cfg: Config, b: float, compare_l4: bool = False) -> dict:
    """Structured backtest result (used by the CLI printer and by the website)."""
    n = len(c)
    split = n // 2
    be = breakeven(b)
    sigs_all = find_signals(c, cfg)
    raw_pop = Population(find_signals(c, replace(cfg, l2_enabled=False)), cfg.spacing, only_l2=False)
    w, m = raw_pop.stats(n, 10**9)
    lo, hi = wilson_interval(w, m)
    res: dict = dict(
        n_candles=n, days=float((c.epoch[-1] - c.epoch[0]) / 86400), payout=b, breakeven=be, split=split,
        raw=dict(n=m, rate=w / m if m else None, lo=lo, hi=hi, proven=bool(m and lo > be)),
        breakeven_table=[dict(payout=p, be=breakeven(p)) for p in (0.70, 0.80, 0.85, 0.90, 0.95)],
        variants=[], l2_check={}, calibration=[])
    for label, vcfg in variants(cfg, compare_l4):
        tr = simulate(c, vcfg, find_signals(c, vcfg), b)
        res["variants"].append(dict(label=label, all=summarize(tr, b),
                                    oos=summarize([t for t in tr if t.idx >= split], b)))
    # Diagnostic snapshot at the final available signal; this is descriptive only and does not alter thresholds.
    epel_diag = None
    if sigs_all:
        epop = Engine(cfg, sigs_all).pop
        last_sig = sigs_all[-1]
        ew, en = epop.stats(last_sig.idx, cfg.estimation_window)
        lr = layer1_epel.evaluate(ew, en, b, cfg)
        epel_diag = dict(candidate_signals=len(sigs_all), resolved_history_n=lr.n, wins=int(ew),
                         p_hat=lr.p_hat if np.isfinite(lr.p_hat) else None,
                         conservative_p=lr.p_cons, breakeven=lr.breakeven,
                         required_conservative_p=lr.breakeven + cfg.min_prob_margin,
                         el_ep_ratio=lr.ratio if np.isfinite(lr.ratio) else None,
                         max_el_ep_ratio=cfg.max_loss_profit_ratio, passed=lr.passed, reason=lr.reason,
                         min_samples=cfg.min_samples, estimation_window=cfg.estimation_window)
    res["epel_diagnostic"] = epel_diag
    h2 = [s for s in sigs_all if s.win is not None and s.idx >= split]
    for name, group in (("passed", [s for s in h2 if s.l2_pass]), ("rejected", [s for s in h2 if not s.l2_pass])):
        if group:
            wn = sum(s.win for s in group)
            l, h = wilson_interval(wn, len(group))
            res["l2_check"][name] = dict(n=len(group), rate=wn / len(group), lo=l, hi=h)
    final = variants(cfg, False)[-1][1]
    tr = simulate(c, final, find_signals(c, final), b)
    edges = [0.0, 0.55, 0.60, 0.65, 0.70, 1.01]
    for a_, z_ in zip(edges[:-1], edges[1:]):
        g = [t for t in tr if a_ <= t.p_cons < z_]
        if g:
            res["calibration"].append(dict(lo=a_, hi=min(z_, 1.0), n=len(g), realised=sum(t.win for t in g) / len(g)))
    return res


def run(c: Candles, cfg: Config, b: float, compare_l4: bool = False) -> None:
    r = analyze(c, cfg, b, compare_l4)
    print(f"\nData: {r['n_candles']} candles, ~{r['days']:.0f} days. Assumed payout b={b:.2f} -> break-even win rate "
          f"{r['breakeven']*100:.1f}% (ties lose). Out-of-sample = second half (candle >= {r['split']}).")
    raw = r["raw"]
    if raw["n"]:
        print(f"\nRAW SIGNAL (thinned, non-overlapping): {raw['n']} signals, hit rate {raw['rate']*100:.1f}%  "
              f"CI95=[{raw['lo']*100:.1f},{raw['hi']*100:.1f}]")
    print("  Break-even by payout:  " + "   ".join(f"{t['payout']:.2f}->{t['be']*100:.1f}%" for t in r["breakeven_table"]))
    if raw["n"] and not raw["proven"]:
        print("  -> lower CI bound is below break-even at the assumed payout: no proven raw edge.")
    print("\nWALK-FORWARD RESULTS (layers added in order; flat $ stake until L3):")
    for v in r["variants"]:
        print(fmt(v["label"] + " [all]", v["all"]))
        print(fmt(v["label"] + " [out-of-sample]", v["oos"]))
    for name, g in r["l2_check"].items():
        print(f"  Filter check, out-of-sample L2 {name:<9} n={g['n']:<5} hit={g['rate']*100:5.1f}%  CI95=[{g['lo']*100:.1f},{g['hi']*100:.1f}]")
    if r["calibration"]:
        print("\nCALIBRATION of the full stack (conservative predicted p vs realised):")
        for g in r["calibration"]:
            print(f"  predicted {g['lo']:.2f}-{g['hi']:.2f}: n={g['n']:<4} realised {g['realised']*100:5.1f}%")
    print("\nReminder: a backtest is not evidence of a future edge. Paper trade before trusting it.\n")


def synthetic(kind: str, n: int = 40000, seed: int = 7) -> Candles:
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, 0.0003, n)
    x = np.zeros(n)
    kappa = {"random": 0.0, "meanrev": 0.06, "strong": 0.35}[kind]
    for i in range(1, n):
        x[i] = x[i - 1] * (1 - kappa) + eps[i]
    close = 1.08 + (np.cumsum(eps) if kind == "random" else x)
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0, 0.0001, n))
    return Candles(epoch=1_700_000_000 + 300 * np.arange(n), open=open_, high=np.maximum(open_, close) + spread,
                   low=np.minimum(open_, close) - spread, close=close)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--synthetic", choices=["random", "meanrev", "strong"])
    ap.add_argument("--payout", type=float, default=0.85, help="assumed net payout per $1 (live uses the real quote)")
    ap.add_argument("--config", default="config.json"); ap.add_argument("--compare-l4", action="store_true")
    a = ap.parse_args()
    cfg = Config.load(a.config)
    if a.synthetic:
        c = synthetic(a.synthetic)
    elif a.csv:
        c = load_csv(a.csv)
    else:
        ap.error("give --csv or --synthetic")
    run(c, cfg, a.payout, a.compare_l4)


if __name__ == "__main__":
    main()
