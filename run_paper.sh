#!/usr/bin/env sh
set -eu
export PAPER_ONLY=1
export BOT_AUTOSTART=signal
exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}"
