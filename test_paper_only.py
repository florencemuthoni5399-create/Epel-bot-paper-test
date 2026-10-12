import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import Config
from data import from_rows
from journal import Journal
from live import Runner
import live


class PaperOnlyTests(unittest.TestCase):
    def test_runner_refuses_demo_mode_even_if_env_tries_to_disable_lock(self):
        # The runner guard is tested at its earliest synchronous validation point.
        with patch.dict(os.environ, {"PAPER_ONLY": "0"}):
            self.assertTrue(live.PAPER_ONLY)
            with self.assertRaisesRegex(RuntimeError, "demo order placement is disabled"):
                asyncio.run(Runner(Config())._session("demo"))

    def test_shadow_settlement_returns_estimated_pnl_for_taken_paper_trade(self):
        with tempfile.TemporaryDirectory() as d:
            j = Journal(str(Path(d) / "journal.db"))
            class Sig:
                epoch = 1000
                direction = "CALL"
                z = 0.0
                er = 0.2
                atr_pct = 50.0
            class Dec:
                l2_pass = True
                action = "TRADE"
                reason = "paper test"
                n = 200
                p_hat = 0.6
                p_cons = 0.55
                breakeven = 0.5
                payout_ratio = 0.8
                ep = 0.1
                el = 0.2
                ratio = 0.5
                stake = 2.0
            j.log(Sig(), Dec(), "signal", 1000)
            # 3 x 5m candles: entry at open[1], settle close[3]. Close rises => CALL win.
            c = from_rows([
                {"epoch":1000,"open":1.0,"high":1.1,"low":0.9,"close":1.0},
                {"epoch":1300,"open":1.0,"high":1.2,"low":0.95,"close":1.05},
                {"epoch":1600,"open":1.05,"high":1.2,"low":1.0,"close":1.1},
                {"epoch":1900,"open":1.1,"high":1.3,"low":1.05,"close":1.2},
            ])
            cfg = Config(granularity=300, hold_candles=3)
            out = j.resolve_shadow(c, cfg)
            self.assertEqual(len(out), 1)
            self.assertAlmostEqual(out[0][1], 1.6)
            self.assertEqual(j.db.execute("SELECT shadow_win FROM decisions").fetchone()[0], 1)
            j.db.close()


if __name__ == "__main__":
    unittest.main()
