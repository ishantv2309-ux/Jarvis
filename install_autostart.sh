#!/usr/bin/env zsh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLIST_TEMPLATE="$PROJECT_DIR/launchd/com.jarvis.assistant.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.jarvis.assistant.plist"

mkdir -p "$HOME/Library/LaunchAgents" "$PROJECT_DIR/logs"
sed "s|__PROJECT_DIR__|$PROJECT_DIR|g" "$PLIST_TEMPLATE" > "$PLIST_DST"

launchctl bootout "gui/$(id -u)/com.jarvis.assistant" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST_DST"
launchctl enable "gui/$(id -u)/com.jarvis.assistant"

echo "Installed and enabled Jarvis login auto-start: $PLIST_DST"
echo "Jarvis will start in low-CPU STANDBY mode and listen for 'Jarvis wake up'."
