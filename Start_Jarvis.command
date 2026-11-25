#!/usr/bin/env zsh
set -euo pipefail

PROJECT_DIR="/Users/ishantverma/Desktop/Jarvis"
cd "$PROJECT_DIR"

# Ensure local Ollama service is active for conversational & multi-step AI
if command -v ollama >/dev/null 2>&1; then
  if ! curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    brew services start ollama >/dev/null 2>&1 || true
  fi
fi

# Stop any older desktop-daemon or stale server on port 8000
pkill -f "jarvis.py" 2>/dev/null || true
if lsof -ti :8000 >/dev/null 2>&1; then
  kill $(lsof -ti :8000) 2>/dev/null || true
  sleep 0.5
fi

echo "Starting Project Jarvis Web Bridge at http://localhost:8000 ..."
"$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/server.py" &
SERVER_PID=$!

# Wait for server readiness, then open browser dashboard
for _ in {1..30}; do
  if curl -s http://127.0.0.1:8000/api/status >/dev/null 2>&1; then
    break
  fi
  sleep 0.4
done

open "http://localhost:8000"
wait "$SERVER_PID"
