#!/usr/bin/env zsh
set -euo pipefail

PROJECT_DIR="/Users/ishantverma/Desktop/Jarvis"
PLIST_SRC="$PROJECT_DIR/launchd/com.user.jarvis.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.user.jarvis.plist"

mkdir -p "$HOME/Library/LaunchAgents" "$PROJECT_DIR/logs"
cp "$PLIST_SRC" "$PLIST_DST"

launchctl bootout "gui/$(id -u)/com.user.jarvis" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST_DST"
launchctl enable "gui/$(id -u)/com.user.jarvis"

echo "Installed and started LaunchAgent: $PLIST_DST"
echo "To stop: launchctl bootout gui/$(id -u)/com.user.jarvis"
