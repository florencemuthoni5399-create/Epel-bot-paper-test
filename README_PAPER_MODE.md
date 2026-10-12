# EP-EL · SMC/ICT Paper Lab — paper-only build

## What this build does
- Connects to Deriv's public WebSocket market-data endpoint.
- Uses the existing causal SMC/ICT-style structural proxy in `strategy.py`: confirmed swing sweep → structure shift → three-candle FVG.
- Evaluates the existing EP-EL / false-signal / risk layers using public payout quotes.
- Journals every candidate, gate decision, and candle-based paper outcome in SQLite (`data/journal.db`).
- Computes estimated paper P&L from the quoted net payout ratio and the candle-based win/loss outcome.
- Uses closed candles only; no real or demo account is needed.

## Safety lock
This package sets `PAPER_ONLY=1`. The dashboard exposes only Stop and Start paper trading. The API, runner, and CLI all refuse demo order placement while the flag is enabled. No order is bought in paper mode.

## Important fidelity limitation
This is an **experimental SMC/ICT structural proxy**, not a faithful deployment of the research paper's complete calibrated-ML + CVaR-LP system. The research package's current dataset is XAU/USD H1, whereas this bot's preserved default is Deriv EUR/USD M5 with a 15-minute binary contract. The research model is not yet trained, validated, and configured for that instrument/timeframe. Do not describe the results as proof that the paper was reproduced or that the strategy has an edge.

## Run locally
```bash
pip install -r requirements.txt
export PAPER_ONLY=1
export BOT_AUTOSTART=signal
uvicorn app:app --host 0.0.0.0 --port 8000
```
Set `DASHBOARD_PASSWORD` for dashboard authentication. No Deriv token is required for paper mode.

## Deploy on Render
Deploy this as a **separate Render service** (the included service name is `epel-smc-ict-paper-lab`) and keep the included `render.yaml` settings. This avoids replacing or interrupting your currently running bot. It sets `BOT_AUTOSTART=signal`, `PAPER_ONLY=1`, and an isolated data directory at `/var/data/smc_ict_paper` so the trial journal does not mix with the existing bot journal; it intentionally omits trading credentials. If your existing Render service is connected to a Git repository, replace its source with these files and deploy. This package does not remotely change or deploy your existing Render service automatically.

## Interpreting results
- Paper P&L is an estimate: it uses a public payout quote at signal time and candle-close outcomes. It does not model every execution detail or slippage.
- Let contracts settle before judging hit rate; pending rows are not results.
- Do not turn off filters or optimize thresholds after inspecting the same paper-trading sample and then call that sample out-of-sample.
- Keep this isolated from the existing bot until the research paper's full ML + LP methodology is faithfully implemented for the target market and passes validation.
