#!/usr/bin/env python3
"""Check required macOS Privacy & Security permissions for Project Jarvis (PRD Section 4)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def check_accessibility() -> bool:
    try:
        from ApplicationServices import AXIsProcessTrusted

        return bool(AXIsProcessTrusted())
    except Exception:
        return False


def check_microphone() -> str:
    try:
        import pyaudio

        pa = pyaudio.PyAudio()
        count = pa.get_device_count()
        default_in = pa.get_default_input_device_info()
        pa.terminate()
        return f"OK (devices={count}, default='{default_in.get('name')}')"
    except Exception as exc:
        return f"WARN ({exc})"


def check_automation_system_events() -> bool:
    proc = subprocess.run(
        ["osascript", "-e", 'tell application "System Events" to get name of first process'],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def check_full_disk_access() -> bool:
    # Safari History or TCC database check is a standard non-destructive FDA indicator
    probe = Path.home() / "Library" / "Safari" / "Bookmarks.plist"
    if not probe.exists():
        probe = Path.home() / "Library" / "Mail"
    try:
        os.listdir(probe) if probe.is_dir() else probe.read_bytes()[:16]
        return True
    except PermissionError:
        return False
    except Exception:
        return True


def main() -> int:
    print("=== Project Jarvis — macOS Permissions Audit ===")
    print(f"1. Microphone        : {check_microphone()}")
    ax = check_accessibility()
    print(f"2. Accessibility     : {'GRANTED' if ax else 'MISSING (System Settings -> Privacy & Security -> Accessibility)'}")
    auto = check_automation_system_events()
    print(f"3. Automation        : {'GRANTED' if auto else 'PROMPT/MISSING (System Settings -> Privacy & Security -> Automation)'}")
    fda = check_full_disk_access()
    print(f"4. Full Disk Access  : {'GRANTED' if fda else 'MISSING (System Settings -> Privacy & Security -> Full Disk Access)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
