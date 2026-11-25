#!/usr/bin/env zsh
set -euo pipefail

PLIST_DST="$HOME/Library/LaunchAgents/com.jarvis.assistant.plist"

launchctl bootout "gui/$(id -u)/com.jarvis.assistant" 2>/dev/null || true
rm -f "$PLIST_DST"

echo "Disabled and removed Jarvis login auto-start ($PLIST_DST)."
