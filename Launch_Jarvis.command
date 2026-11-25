#!/usr/bin/env zsh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/logs"

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

# Kill any stale process holding port 8000 before starting Uvicorn
STALE_PIDS="$(lsof -ti:8000 2>/dev/null || true)"
if [[ -n "$STALE_PIDS" ]]; then
  echo "Cleaning up stale processes on port 8000..."
  echo "$STALE_PIDS" | xargs kill -9 2>/dev/null || true
  sleep 0.5
fi

# Ensure local Ollama is running
if command -v ollama >/dev/null 2>&1; then
  if ! curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    brew services start ollama >/dev/null 2>&1 || nohup ollama serve >"$PROJECT_DIR/logs/ollama.log" 2>&1 &
  fi
fi

# Activate local virtual environment
source "$PROJECT_DIR/.venv/bin/activate"

echo "Starting J.A.R.V.I.S. (ACTIVE LISTENING) at http://localhost:8000 ..."
python "$PROJECT_DIR/server.py" --open-browser
