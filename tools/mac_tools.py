"""macOS System Execution & YouTube Media Automation Registry (FR-2 & FR-3).

Exposes structured tool definitions and execution methods for:
- Application lifecycle (NSWorkspace + open -a + AppleScript)
- Hardware & OS settings (volume, brightness, Wi-Fi, Bluetooth, Dark Mode, DND)
- File search (Spotlight mdfind), file management, and terminal execution
- Screen awareness & UI automation (AXUIElement Accessibility API + pyautogui)
- YouTube direct search & auto-play (yt-dlp + browser JS injection)
- Browser media playback controls (Chrome/Safari AppleScript JS + media keys)
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from core.safety import SafetyGuard

logger = logging.getLogger(__name__)

_APP_ALIASES: dict[str, str] = {
    "vs code": "Visual Studio Code",
    "vscode": "Visual Studio Code",
    "code": "Visual Studio Code",
    "chrome": "Google Chrome",
    "google chrome": "Google Chrome",
    "safari": "Safari",
    "finder": "Finder",
    "terminal": "Terminal",
    "iterm": "iTerm",
    "iterm2": "iTerm",
    "notes": "Notes",
    "music": "Music",
    "spotify": "Spotify",
    "slack": "Slack",
    "mail": "Mail",
    "messages": "Messages",
    "settings": "System Settings",
    "system settings": "System Settings",
    "system preferences": "System Settings",
    "activity monitor": "Activity Monitor",
    "calculator": "Calculator",
    "calendar": "Calendar",
    "preview": "Preview",
}


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    tool_name: str
    message: str
    data: dict[str, Any] | None = None
    latency_ms: float = 0.0


def _run_cmd(cmd: list[str], timeout: float = 15.0) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _run_osascript(script: str, timeout: float = 10.0) -> tuple[int, str, str]:
    return _run_cmd(["osascript", "-e", script], timeout=timeout)


def _escape_applescript(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def resolve_app_name(name: str) -> str:
    """Resolve spoken/short application names to installed macOS .app bundle names."""
    cleaned = name.strip()
    lowered = cleaned.lower()
    if lowered in _APP_ALIASES:
        return _APP_ALIASES[lowered]

    # Scan standard macOS application directories for exact/fuzzy match
    search_dirs = [
        Path("/Applications"),
        Path("/System/Applications"),
        Path("/System/Applications/Utilities"),
        Path.home() / "Applications",
    ]
    candidates: list[str] = []
    for root in search_dirs:
        if not root.exists():
            continue
        try:
            for item in root.iterdir():
                if item.suffix == ".app":
                    candidates.append(item.stem)
        except PermissionError:
            continue

    for c in candidates:
        if c.lower() == lowered:
            return c
    for c in candidates:
        if lowered in c.lower():
            return c
    return cleaned


def _send_macos_media_key(key_type: int) -> bool:
    """Synthesize a macOS NX media key press/release via Quartz NSEvent."""
    try:
        import AppKit
        import Quartz

        for flags in (0xA00, 0xB00):  # keydown (0xA), keyup (0xB)
            ev = AppKit.NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
                AppKit.NSSystemDefined,
                (0, 0),
                flags,
                0,
                0,
                None,
                8,
                (key_type << 16) | flags,
                -1,
            )
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev.CGEvent())
        return True
    except Exception as exc:
        logger.debug("Media key synthesis failed: %s", exc)
        return False


# Tool schema declarations for Gemini / Ollama / internal dispatch
TOOL_DECLARATIONS: list[dict[str, Any]] = [
    {
        "name": "manage_application",
        "description": "Open, focus, minimize, hide, quit, force-quit, or list macOS applications.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["open", "focus", "minimize", "hide", "quit", "force_quit", "list_running"],
                },
                "app_name": {"type": "string", "description": "Application name (e.g. 'VS Code', 'Safari')."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "control_os_setting",
        "description": "Control macOS system settings: volume, brightness, wifi, bluetooth, dark_mode, do_not_disturb.",
        "parameters": {
            "type": "object",
            "properties": {
                "setting": {
                    "type": "string",
                    "enum": ["volume", "brightness", "wifi", "bluetooth", "dark_mode", "do_not_disturb"],
                },
                "value": {
                    "type": "string",
                    "description": "Target value: 'on'/'off'/'toggle', 'up'/'down'/'mute'/'unmute', or 0-100.",
                },
            },
            "required": ["setting", "value"],
        },
    },
    {
        "name": "search_files",
        "description": "Search local files using macOS Spotlight (mdfind).",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "directory": {"type": "string", "description": "Optional folder path to restrict search."},
                "limit": {"type": "integer", "description": "Maximum results (default 10)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "manage_files",
        "description": "List, create directories, move, copy, or organize files by extension.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["list", "mkdir", "move", "copy", "organize_by_type"]},
                "source": {"type": "string"},
                "destination": {"type": "string"},
            },
            "required": ["action", "source"],
        },
    },
    {
        "name": "delete_files",
        "description": "Delete files or move them to Trash (DESTRUCTIVE: requires spoken 'Yes, execute').",
        "parameters": {
            "type": "object",
            "properties": {
                "paths": {"type": "array", "items": {"type": "string"}},
                "permanent": {"type": "boolean", "description": "If true, permanently rm instead of Trash."},
            },
            "required": ["paths"],
        },
    },
    {
        "name": "run_shell_command",
        "description": "Execute a macOS zsh shell command. Destructive commands require spoken confirmation.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "run_applescript",
        "description": "Run AppleScript via osascript.",
        "parameters": {
            "type": "object",
            "properties": {
                "script": {"type": "string"},
            },
            "required": ["script"],
        },
    },
    {
        "name": "get_active_window",
        "description": "Read frontmost application name, active window title, and visible UI elements via AXUIElement.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "click_ui_element",
        "description": "Click a UI element by accessibility label/title via AXUIElement or by (x, y) coordinates.",
        "parameters": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "description": "Accessible title/description of element to click."},
                "x": {"type": "integer", "description": "Screen X coordinate."},
                "y": {"type": "integer", "description": "Screen Y coordinate."},
            },
        },
    },
    {
        "name": "youtube_search_and_play",
        "description": "Search YouTube for a query, resolve the top video URL, open in browser and autoplay.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "browser": {"type": "string", "description": "'default', 'Google Chrome', or 'Safari'."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "media_control",
        "description": "Control YouTube/browser/media playback: pause, resume, skip_ad, mute, unmute, toggle_mute, fullscreen, next_video, previous_video.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "pause",
                        "resume",
                        "play",
                        "toggle",
                        "skip_ad",
                        "mute",
                        "unmute",
                        "toggle_mute",
                        "fullscreen",
                        "next_video",
                        "previous_video",
                    ],
                },
                "browser": {"type": "string", "description": "'auto', 'Google Chrome', or 'Safari'."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "system_power",
        "description": "Shut down, restart, or sleep Mac (DESTRUCTIVE: requires spoken 'Yes, execute').",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["shutdown", "restart", "sleep"]},
            },
            "required": ["action"],
        },
    },
    {
        "name": "send_email",
        "description": "Send an email via Apple Mail (DESTRUCTIVE: requires spoken 'Yes, execute').",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["recipient", "subject", "body"],
        },
    },
    {
        "name": "send_message",
        "description": "Send an iMessage/SMS via Apple Messages (DESTRUCTIVE: requires spoken 'Yes, execute').",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["recipient", "body"],
        },
    },
    {
        "name": "whatsapp_send_message",
        "description": "Send a WhatsApp Desktop message to a contact name or phone number.",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {"type": "string", "description": "Contact name/nickname or phone number."},
                "message": {"type": "string", "description": "Message text to send."},
            },
            "required": ["recipient", "message"],
        },
    },
    {
        "name": "whatsapp_call",
        "description": "Place a WhatsApp Desktop voice or video call to a contact.",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {"type": "string", "description": "Contact name/nickname or phone number."},
                "call_type": {"type": "string", "enum": ["voice", "video"]},
            },
            "required": ["recipient"],
        },
    },
    {
        "name": "whatsapp_answer_call",
        "description": "Answer an incoming WhatsApp Desktop voice or video call.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "whatsapp_decline_call",
        "description": "Decline/reject an incoming WhatsApp Desktop voice or video call.",
        "parameters": {"type": "object", "properties": {}},
    },
]


class MacToolRegistry:
    """Executes structured macOS system & media automation tools with safety guardrails."""

    def __init__(self, safety_guard: Optional[SafetyGuard] = None) -> None:
        self.safety_guard = safety_guard or SafetyGuard()
        self._handlers: dict[str, Callable[..., ToolResult]] = {
            "manage_application": self.manage_application,
            "control_os_setting": self.control_os_setting,
            "search_files": self.search_files,
            "manage_files": self.manage_files,
            "delete_files": self.delete_files,
            "run_shell_command": self.run_shell_command,
            "run_applescript": self.run_applescript,
            "get_active_window": self.get_active_window,
            "click_ui_element": self.click_ui_element,
            "youtube_search_and_play": self.youtube_search_and_play,
            "media_control": self.media_control,
            "system_power": self.system_power,
            "send_email": self.send_email,
            "send_message": self.send_message,
            "whatsapp_send_message": self.whatsapp_send_message,
            "whatsapp_call": self.whatsapp_call,
            "whatsapp_answer_call": self.whatsapp_answer_call,
            "whatsapp_decline_call": self.whatsapp_decline_call,
        }

    def execute(self, tool_name: str, args: dict[str, Any] | None = None) -> ToolResult:
        """Authorize and execute a named tool with structured arguments."""
        t0 = time.perf_counter()
        payload = args or {}
        handler = self._handlers.get(tool_name)
        if handler is None:
            return ToolResult(
                ok=False,
                tool_name=tool_name,
                message=f"Unknown tool: {tool_name}",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        allowed, reason = self.safety_guard.authorize(tool_name, payload)
        if not allowed:
            return ToolResult(
                ok=False,
                tool_name=tool_name,
                message=reason,
                data={"blocked_by_safety": True},
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        try:
            result = handler(**payload)
            elapsed = (time.perf_counter() - t0) * 1000.0
            return ToolResult(
                ok=result.ok,
                tool_name=result.tool_name,
                message=result.message,
                data=result.data,
                latency_ms=elapsed,
            )
        except TypeError as exc:
            return ToolResult(
                ok=False,
                tool_name=tool_name,
                message=f"Invalid arguments for {tool_name}: {exc}",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )
        except Exception as exc:
            logger.exception("Tool '%s' execution failed", tool_name)
            return ToolResult(
                ok=False,
                tool_name=tool_name,
                message=f"Execution error in {tool_name}: {exc}",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

    # ------------------------------------------------------------------
    # 1. Application Lifecycle (NSWorkspace + open -a + AppleScript)
    # ------------------------------------------------------------------
    def manage_application(self, action: str, app_name: str = "") -> ToolResult:
        action = action.lower().strip()
        if action == "list_running":
            from AppKit import NSWorkspace

            apps = [
                a.localizedName()
                for a in NSWorkspace.sharedWorkspace().runningApplications()
                if a.activationPolicy() == 0 and a.localizedName()
            ]
            return ToolResult(
                ok=True,
                tool_name="manage_application",
                message=f"Running apps: {', '.join(apps)}",
                data={"apps": apps},
            )

        if not app_name:
            return ToolResult(ok=False, tool_name="manage_application", message="app_name is required.")

        resolved = resolve_app_name(app_name)

        if action == "open":
            from AppKit import NSWorkspace

            ws = NSWorkspace.sharedWorkspace()
            # Fast-path: check if already running to focus immediately, else `open -a`
            rc, _, err = _run_cmd(["open", "-a", resolved], timeout=8.0)
            if rc != 0:
                launched = ws.launchApplication_(resolved)
                if not launched:
                    return ToolResult(
                        ok=False,
                        tool_name="manage_application",
                        message=f"Could not open {resolved}: {err}",
                    )
            return ToolResult(
                ok=True,
                tool_name="manage_application",
                message=f"Opened {resolved}.",
                data={"app": resolved},
            )

        if action == "focus":
            from AppKit import NSApplicationActivateIgnoringOtherApps, NSWorkspace

            for app in NSWorkspace.sharedWorkspace().runningApplications():
                name = app.localizedName() or ""
                if name.lower() == resolved.lower() or resolved.lower() in name.lower():
                    app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
                    return ToolResult(
                        ok=True,
                        tool_name="manage_application",
                        message=f"Focused {name}.",
                        data={"app": name},
                    )
            # Launch if not running yet
            rc, _, err = _run_cmd(["open", "-a", resolved], timeout=8.0)
            return ToolResult(
                ok=(rc == 0),
                tool_name="manage_application",
                message=f"Focused {resolved}." if rc == 0 else f"Failed to focus {resolved}: {err}",
            )

        if action == "minimize":
            esc = _escape_applescript(resolved)
            script = (
                f'tell application "System Events" to tell process "{esc}" '
                'to set value of attribute "AXMinimized" of window 1 to true'
            )
            rc, _, err = _run_osascript(script)
            if rc != 0:
                # Fallback via app's own AppleScript dictionary
                rc, _, err = _run_osascript(
                    f'tell application "{esc}" to set miniaturized of window 1 to true'
                )
            return ToolResult(
                ok=(rc == 0),
                tool_name="manage_application",
                message=f"Minimized {resolved}." if rc == 0 else f"Could not minimize {resolved}: {err}",
            )

        if action == "hide":
            from AppKit import NSWorkspace

            for app in NSWorkspace.sharedWorkspace().runningApplications():
                if (app.localizedName() or "").lower() == resolved.lower():
                    app.hide()
                    return ToolResult(ok=True, tool_name="manage_application", message=f"Hid {resolved}.")
            return ToolResult(ok=False, tool_name="manage_application", message=f"{resolved} is not running.")

        if action in ("quit", "force_quit"):
            from AppKit import NSWorkspace

            matched = False
            for app in NSWorkspace.sharedWorkspace().runningApplications():
                name = app.localizedName() or ""
                if name.lower() == resolved.lower() or resolved.lower() in name.lower():
                    matched = True
                    ok = app.forceTerminate() if action == "force_quit" else app.terminate()
                    if not ok and action == "quit":
                        _run_osascript(f'tell application "{_escape_applescript(name)}" to quit')
            if matched:
                verb = "Force-quit" if action == "force_quit" else "Quit"
                return ToolResult(ok=True, tool_name="manage_application", message=f"{verb} {resolved}.")
            return ToolResult(ok=False, tool_name="manage_application", message=f"{resolved} is not running.")

        return ToolResult(ok=False, tool_name="manage_application", message=f"Unsupported action: {action}")

    # ------------------------------------------------------------------
    # 2. Hardware & OS Settings
    # ------------------------------------------------------------------
    def control_os_setting(self, setting: str, value: str) -> ToolResult:
        setting = setting.lower().strip()
        val = str(value).strip().lower()

        if setting == "volume":
            if val in ("mute", "muted"):
                rc, _, err = _run_osascript("set volume with output muted")
                return ToolResult(ok=(rc == 0), tool_name="control_os_setting", message="Volume muted." if rc == 0 else err)
            if val in ("unmute", "unmuted"):
                rc, _, err = _run_osascript("set volume without output muted")
                return ToolResult(ok=(rc == 0), tool_name="control_os_setting", message="Volume unmuted." if rc == 0 else err)
            if val in ("up", "increase"):
                rc, _, err = _run_osascript(
                    "set cur to output volume of (get volume settings)\n"
                    "set volume output volume (cur + 15)"
                )
                return ToolResult(ok=(rc == 0), tool_name="control_os_setting", message="Increased volume." if rc == 0 else err)
            if val in ("down", "decrease"):
                rc, _, err = _run_osascript(
                    "set cur to output volume of (get volume settings)\n"
                    "set volume output volume (cur - 15)"
                )
                return ToolResult(ok=(rc == 0), tool_name="control_os_setting", message="Decreased volume." if rc == 0 else err)
            num_match = re.search(r"\d+", val)
            if num_match:
                level = max(0, min(100, int(num_match.group())))
                rc, _, err = _run_osascript(f"set volume output volume {level}")
                return ToolResult(ok=(rc == 0), tool_name="control_os_setting", message=f"Volume set to {level}%." if rc == 0 else err)
            return ToolResult(ok=False, tool_name="control_os_setting", message=f"Invalid volume value: {value}")

        if setting == "dark_mode":
            if val == "on":
                script = 'tell application "System Events" to tell appearance preferences to set dark mode to true'
            elif val == "off":
                script = 'tell application "System Events" to tell appearance preferences to set dark mode to false'
            else:
                script = 'tell application "System Events" to tell appearance preferences to set dark mode to not dark mode'
            rc, _, err = _run_osascript(script)
            return ToolResult(
                ok=(rc == 0),
                tool_name="control_os_setting",
                message=f"Dark Mode updated ({val})." if rc == 0 else f"Dark Mode failed: {err}",
            )

        if setting == "wifi":
            power = "on" if val in ("on", "true", "enable") else "off"
            iface = self._detect_wifi_interface()
            rc, _, err = _run_cmd(["networksetup", "-setairportpower", iface, power])
            return ToolResult(
                ok=(rc == 0),
                tool_name="control_os_setting",
                message=f"Wi-Fi turned {power}." if rc == 0 else f"Wi-Fi error: {err}",
            )

        if setting == "bluetooth":
            target = "1" if val in ("on", "true", "enable") else "0"
            if shutil.which("blueutil"):
                rc, _, err = _run_cmd(["blueutil", "--power", target])
                return ToolResult(
                    ok=(rc == 0),
                    tool_name="control_os_setting",
                    message=f"Bluetooth turned {'on' if target == '1' else 'off'}." if rc == 0 else err,
                )
            shortcut_name = "Turn Bluetooth On" if target == "1" else "Turn Bluetooth Off"
            rc, _, err = _run_cmd(["shortcuts", "run", shortcut_name])
            return ToolResult(
                ok=(rc == 0),
                tool_name="control_os_setting",
                message=f"Bluetooth updated via Shortcuts." if rc == 0 else (
                    f"Bluetooth control requires `brew install blueutil` or a '{shortcut_name}' macOS Shortcut ({err})."
                ),
            )

        if setting == "do_not_disturb":
            shortcut_name = "Turn Focus On" if val in ("on", "true", "enable") else "Turn Focus Off"
            rc, _, err = _run_cmd(["shortcuts", "run", shortcut_name])
            if rc != 0:
                # Try generic Do Not Disturb shortcut names
                alt = "Do Not Disturb On" if val in ("on", "true", "enable") else "Do Not Disturb Off"
                rc, _, err = _run_cmd(["shortcuts", "run", alt])
            return ToolResult(
                ok=(rc == 0),
                tool_name="control_os_setting",
                message=f"Do Not Disturb updated ({val})." if rc == 0 else (
                    f"Create a macOS Shortcut named '{shortcut_name}' to toggle Focus via CLI ({err})."
                ),
            )

        if setting == "brightness":
            if shutil.which("brightness"):
                if val in ("up", "increase"):
                    rc, out, _ = _run_cmd(["brightness", "-l"])
                    cur = 0.5
                    m = re.search(r"brightness\s+([0-9.]+)", out)
                    if m:
                        cur = float(m.group(1))
                    rc, _, err = _run_cmd(["brightness", f"{min(1.0, cur + 0.15):.2f}"])
                elif val in ("down", "decrease"):
                    rc, out, _ = _run_cmd(["brightness", "-l"])
                    cur = 0.5
                    m = re.search(r"brightness\s+([0-9.]+)", out)
                    if m:
                        cur = float(m.group(1))
                    rc, _, err = _run_cmd(["brightness", f"{max(0.05, cur - 0.15):.2f}"])
                else:
                    num = re.search(r"\d+", val)
                    lvl = max(0, min(100, int(num.group()))) / 100.0 if num else 0.6
                    rc, _, err = _run_cmd(["brightness", f"{lvl:.2f}"])
                return ToolResult(
                    ok=(rc == 0),
                    tool_name="control_os_setting",
                    message="Brightness updated." if rc == 0 else err,
                )
            # AppleScript key code fallback (144 = Brightness Up, 145 = Brightness Down)
            steps = 3
            code = 144 if val in ("up", "increase") else 145
            if num := re.search(r"\d+", val):
                target_pct = max(0, min(100, int(num.group())))
                code = 144 if target_pct >= 50 else 145
            script = (
                'tell application "System Events"\n'
                + f"repeat {steps} times\nkey code {code}\nend repeat\n"
                + "end tell"
            )
            rc, _, err = _run_osascript(script)
            return ToolResult(
                ok=(rc == 0),
                tool_name="control_os_setting",
                message="Adjusted screen brightness." if rc == 0 else err,
            )

        return ToolResult(ok=False, tool_name="control_os_setting", message=f"Unknown setting: {setting}")

    @staticmethod
    def _detect_wifi_interface() -> str:
        rc, out, _ = _run_cmd(["networksetup", "-listallhardwareports"])
        if rc == 0:
            match = re.search(
                r"Hardware Port:\s*(?:Wi-Fi|AirPort)\s*\nDevice:\s*(\S+)", out
            )
            if match:
                return match.group(1)
        return "en0"

    # ------------------------------------------------------------------
    # 3. File System & Terminal Execution
    # ------------------------------------------------------------------
    def search_files(self, query: str, directory: str = "", limit: int = 10) -> ToolResult:
        cmd = ["mdfind"]
        if directory:
            scope = str(Path(directory).expanduser())
            cmd.extend(["-onlyin", scope])
        cmd.append(query)
        rc, out, err = _run_cmd(cmd, timeout=10.0)
        if rc != 0:
            return ToolResult(ok=False, tool_name="search_files", message=f"Spotlight search failed: {err}")
        lines = [ln for ln in out.splitlines() if ln.strip()][: max(1, int(limit))]
        return ToolResult(
            ok=True,
            tool_name="search_files",
            message=f"Found {len(lines)} matching file(s).",
            data={"files": lines},
        )

    def manage_files(self, action: str, source: str, destination: str = "") -> ToolResult:
        action = action.lower().strip()
        src = Path(source).expanduser()

        if action == "list":
            if not src.exists():
                return ToolResult(ok=False, tool_name="manage_files", message=f"Path not found: {src}")
            items = sorted(p.name for p in src.iterdir())
            return ToolResult(
                ok=True,
                tool_name="manage_files",
                message=f"Listed {len(items)} items in {src.name}.",
                data={"items": items[:50]},
            )

        if action == "mkdir":
            src.mkdir(parents=True, exist_ok=True)
            return ToolResult(ok=True, tool_name="manage_files", message=f"Created directory {src}.")

        if action == "move":
            if not destination:
                return ToolResult(ok=False, tool_name="manage_files", message="destination is required for move.")
            dst = Path(destination).expanduser()
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            return ToolResult(ok=True, tool_name="manage_files", message=f"Moved {src.name} to {dst}.")

        if action == "copy":
            if not destination:
                return ToolResult(ok=False, tool_name="manage_files", message="destination is required for copy.")
            dst = Path(destination).expanduser()
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
            return ToolResult(ok=True, tool_name="manage_files", message=f"Copied {src.name} to {dst}.")

        if action == "organize_by_type":
            if not src.is_dir():
                return ToolResult(ok=False, tool_name="manage_files", message=f"Not a directory: {src}")
            categories = {
                "Images": {".png", ".jpg", ".jpeg", ".gif", ".webp", ".heic", ".svg"},
                "Documents": {".pdf", ".docx", ".doc", ".txt", ".md", ".xlsx", ".pptx", ".csv"},
                "Audio": {".mp3", ".wav", ".m4a", ".flac", ".aac"},
                "Video": {".mp4", ".mov", ".mkv", ".avi"},
                "Archives": {".zip", ".tar", ".gz", ".dmg", ".pkg"},
            }
            moved = 0
            for item in src.iterdir():
                if not item.is_file() or item.name.startswith("."):
                    continue
                ext = item.suffix.lower()
                folder = "Other"
                for cat, exts in categories.items():
                    if ext in exts:
                        folder = cat
                        break
                target_dir = src / folder
                target_dir.mkdir(exist_ok=True)
                shutil.move(str(item), str(target_dir / item.name))
                moved += 1
            return ToolResult(
                ok=True,
                tool_name="manage_files",
                message=f"Organized {moved} files in {src.name}.",
                data={"moved_count": moved},
            )

        return ToolResult(ok=False, tool_name="manage_files", message=f"Unknown file action: {action}")

    def delete_files(self, paths: list[str], permanent: bool = False) -> ToolResult:
        deleted: list[str] = []
        for raw_p in paths:
            p = Path(raw_p).expanduser()
            if not p.exists():
                continue
            if permanent:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
            else:
                from AppKit import NSURL, NSFileManager

                url = NSURL.fileURLWithPath_(str(p))
                ok, _, err = NSFileManager.defaultManager().trashItemAtURL_resultingItemURL_error_(
                    url, None, None
                )
                if not ok:
                    return ToolResult(
                        ok=False,
                        tool_name="delete_files",
                        message=f"Could not move {p} to Trash: {err}",
                    )
            deleted.append(str(p))
        return ToolResult(
            ok=True,
            tool_name="delete_files",
            message=f"Deleted {len(deleted)} item(s).",
            data={"deleted": deleted},
        )

    def run_shell_command(self, command: str) -> ToolResult:
        proc = subprocess.run(
            ["/bin/zsh", "-lc", command],
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
        output = proc.stdout.strip() or proc.stderr.strip()
        return ToolResult(
            ok=(proc.returncode == 0),
            tool_name="run_shell_command",
            message=output[:500] if output else f"Command exited with code {proc.returncode}.",
            data={"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr},
        )

    def run_applescript(self, script: str) -> ToolResult:
        rc, out, err = _run_osascript(script, timeout=15.0)
        return ToolResult(
            ok=(rc == 0),
            tool_name="run_applescript",
            message=out or err or ("AppleScript executed." if rc == 0 else "AppleScript failed."),
            data={"returncode": rc, "stdout": out, "stderr": err},
        )

    # ------------------------------------------------------------------
    # 4. Screen Awareness & UI Automation (AXUIElement + pyautogui)
    # ------------------------------------------------------------------
    def get_active_window(self) -> ToolResult:
        from AppKit import NSWorkspace
        from ApplicationServices import (
            AXUIElementCopyAttributeValue,
            AXUIElementCreateApplication,
            kAXChildrenAttribute,
            kAXFocusedWindowAttribute,
            kAXTitleAttribute,
        )

        front = NSWorkspace.sharedWorkspace().frontmostApplication()
        if front is None:
            return ToolResult(ok=False, tool_name="get_active_window", message="No frontmost application.")

        app_name = front.localizedName() or "Unknown"
        pid = front.processIdentifier()
        app_ref = AXUIElementCreateApplication(pid)

        err, win_ref = AXUIElementCopyAttributeValue(app_ref, kAXFocusedWindowAttribute, None)
        win_title = ""
        elements: list[str] = []
        if err == 0 and win_ref is not None:
            t_err, title_val = AXUIElementCopyAttributeValue(win_ref, kAXTitleAttribute, None)
            if t_err == 0 and title_val:
                win_title = str(title_val)
            elements = self._collect_ax_labels(win_ref, max_depth=3, max_items=25)

        return ToolResult(
            ok=True,
            tool_name="get_active_window",
            message=f"Active app is {app_name}" + (f", window '{win_title}'" if win_title else "."),
            data={"app": app_name, "pid": pid, "window_title": win_title, "ui_elements": elements},
        )

    def _collect_ax_labels(self, element, max_depth: int = 3, max_items: int = 25) -> list[str]:
        from ApplicationServices import (
            AXUIElementCopyAttributeValue,
            kAXChildrenAttribute,
            kAXDescriptionAttribute,
            kAXTitleAttribute,
        )

        labels: list[str] = []
        queue = [(element, 0)]
        while queue and len(labels) < max_items:
            node, depth = queue.pop(0)
            for attr in (kAXTitleAttribute, kAXDescriptionAttribute):
                err, val = AXUIElementCopyAttributeValue(node, attr, None)
                if err == 0 and val and isinstance(val, str) and val.strip():
                    labels.append(val.strip())
                    break
            if depth < max_depth:
                c_err, children = AXUIElementCopyAttributeValue(node, kAXChildrenAttribute, None)
                if c_err == 0 and children:
                    for ch in list(children)[:30]:
                        queue.append((ch, depth + 1))
        return labels

    def click_ui_element(
        self,
        label: str = "",
        x: Optional[int] = None,
        y: Optional[int] = None,
    ) -> ToolResult:
        if x is not None and y is not None:
            import pyautogui

            pyautogui.click(int(x), int(y))
            return ToolResult(
                ok=True,
                tool_name="click_ui_element",
                message=f"Clicked screen coordinates ({x}, {y}).",
            )

        if not label:
            return ToolResult(ok=False, tool_name="click_ui_element", message="Provide label or (x, y) coordinates.")

        from AppKit import NSWorkspace
        from ApplicationServices import (
            AXUIElementCopyAttributeValue,
            AXUIElementCreateApplication,
            AXUIElementPerformAction,
            kAXChildrenAttribute,
            kAXDescriptionAttribute,
            kAXPressAction,
            kAXTitleAttribute,
        )

        front = NSWorkspace.sharedWorkspace().frontmostApplication()
        if front is None:
            return ToolResult(ok=False, tool_name="click_ui_element", message="No frontmost app.")

        app_ref = AXUIElementCreateApplication(front.processIdentifier())
        target = label.lower().strip()
        queue = [(app_ref, 0)]
        visited = 0
        while queue and visited < 400:
            node, depth = queue.pop(0)
            visited += 1
            for attr in (kAXTitleAttribute, kAXDescriptionAttribute):
                err, val = AXUIElementCopyAttributeValue(node, attr, None)
                if err == 0 and isinstance(val, str) and target in val.lower():
                    act_err = AXUIElementPerformAction(node, kAXPressAction)
                    if act_err == 0:
                        return ToolResult(
                            ok=True,
                            tool_name="click_ui_element",
                            message=f"Pressed UI element '{val}'.",
                            data={"matched_label": val},
                        )
            if depth < 5:
                c_err, children = AXUIElementCopyAttributeValue(node, kAXChildrenAttribute, None)
                if c_err == 0 and children:
                    for ch in list(children)[:40]:
                        queue.append((ch, depth + 1))

        return ToolResult(
            ok=False,
            tool_name="click_ui_element",
            message=f"Could not find clickable UI element matching '{label}'.",
        )

    # ------------------------------------------------------------------
    # 5. YouTube & Browser Media Automation (FR-3)
    # ------------------------------------------------------------------
    @staticmethod
    def resolve_youtube_top_url(query: str) -> tuple[str, str]:
        """Resolve the top YouTube video URL and title for a search query via yt-dlp."""
        try:
            import yt_dlp

            opts = {
                "quiet": True,
                "no_warnings": True,
                "extract_flat": True,
                "skip_download": True,
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(f"ytsearch1:{query}", download=False)
                entries = (info or {}).get("entries") or []
                if entries:
                    top = entries[0]
                    vid_id = top.get("id")
                    title = top.get("title") or query
                    url = top.get("url") or (f"https://www.youtube.com/watch?v={vid_id}" if vid_id else "")
                    if url and not url.startswith("http"):
                        url = f"https://www.youtube.com/watch?v={url}"
                    if url:
                        return url, title
        except Exception as exc:
            logger.warning("yt-dlp YouTube resolution failed (%s); falling back to search URL.", exc)

        encoded = urllib.parse.quote_plus(query)
        return f"https://www.youtube.com/results?search_query={encoded}", query

    def youtube_search_and_play(self, query: str, browser: str = "default") -> ToolResult:
        url, title = self.resolve_youtube_top_url(query)
        sep = "&" if "?" in url else "?"
        autoplay_url = f"{url}{sep}autoplay=1" if "watch?v=" in url else url

        if browser.lower() in ("default", ""):
            rc, _, err = _run_cmd(["open", autoplay_url], timeout=8.0)
        else:
            resolved_browser = resolve_app_name(browser)
            rc, _, err = _run_cmd(["open", "-a", resolved_browser, autoplay_url], timeout=8.0)

        if rc != 0:
            return ToolResult(
                ok=False,
                tool_name="youtube_search_and_play",
                message=f"Failed to open YouTube video: {err}",
            )

        # Ensure playback starts even if browser blocks unmuted autoplay
        time.sleep(1.5)
        self._inject_youtube_js(
            "var v=document.querySelector('video'); if(v && v.paused){v.play().catch(()=>{});}",
            browser=browser,
        )

        return ToolResult(
            ok=True,
            tool_name="youtube_search_and_play",
            message=f"Playing {title} on YouTube.",
            data={"url": autoplay_url, "title": title},
        )

    def media_control(self, action: str, browser: str = "auto") -> ToolResult:
        act = action.lower().strip()
        js_map = {
            "pause": "var v=document.querySelector('video'); if(v){v.pause(); 'paused';}",
            "resume": "var v=document.querySelector('video'); if(v){v.play(); 'playing';}",
            "play": "var v=document.querySelector('video'); if(v){v.play(); 'playing';}",
            "toggle": "var v=document.querySelector('video'); if(v){if(v.paused){v.play();'playing';}else{v.pause();'paused';}}",
            "mute": "var v=document.querySelector('video'); if(v){v.muted=true; 'muted';}",
            "unmute": "var v=document.querySelector('video'); if(v){v.muted=false; 'unmuted';}",
            "toggle_mute": "var v=document.querySelector('video'); if(v){v.muted=!v.muted; v.muted?'muted':'unmuted';}",
            "skip_ad": (
                "var b=document.querySelector('.ytp-skip-ad-button, .ytp-ad-skip-button, "
                ".ytp-ad-skip-button-modern'); if(b){b.click(); 'ad_skipped';} else {'no_ad';}"
            ),
            "fullscreen": (
                "var b=document.querySelector('.ytp-fullscreen-button'); "
                "if(b){b.click(); 'fullscreen_toggled';}"
            ),
            "next_video": (
                "var b=document.querySelector('.ytp-next-button'); "
                "if(b){b.click(); 'next_video';}"
            ),
            "previous_video": "window.history.back(); 'previous_video';",
        }

        js = js_map.get(act)
        if js is None:
            return ToolResult(ok=False, tool_name="media_control", message=f"Unknown media action: {action}")

        ok, out = self._inject_youtube_js(js, browser=browser)
        if ok and out:
            return ToolResult(
                ok=True,
                tool_name="media_control",
                message=f"Media action {act}: {out}.",
                data={"result": out},
            )

        # Fallback to macOS hardware media key events (NX_KEYTYPE_PLAY=16, NEXT=17, PREV=18)
        nx_map = {"pause": 16, "resume": 16, "play": 16, "toggle": 16, "next_video": 17, "previous_video": 18}
        if act in nx_map and _send_macos_media_key(nx_map[act]):
            return ToolResult(
                ok=True,
                tool_name="media_control",
                message=f"Sent macOS media key for {act}.",
            )

        return ToolResult(
            ok=False,
            tool_name="media_control",
            message=f"Could not perform media action '{act}' (no active YouTube tab or media key access).",
        )

    @staticmethod
    def _inject_youtube_js(js: str, browser: str = "auto") -> tuple[bool, str]:
        target_browsers = (
            [resolve_app_name(browser)]
            if browser.lower() not in ("auto", "default", "")
            else ["Google Chrome", "Safari"]
        )
        esc_js = _escape_applescript(js)

        for b in target_browsers:
            if b == "Google Chrome":
                script = (
                    'if application "Google Chrome" is running then\n'
                    'tell application "Google Chrome" to execute active tab of front window '
                    f'javascript "{esc_js}"\n'
                    "end if"
                )
            elif b == "Safari":
                script = (
                    'if application "Safari" is running then\n'
                    f'tell application "Safari" to do JavaScript "{esc_js}" '
                    "in current tab of front window\n"
                    "end if"
                )
            else:
                continue
            rc, out, _ = _run_osascript(script, timeout=5.0)
            if rc == 0 and out:
                return True, out
        return False, ""

    # ------------------------------------------------------------------
    # 6. Destructive Actions (Gated by SafetyGuard)
    # ------------------------------------------------------------------
    def system_power(self, action: str) -> ToolResult:
        act = action.lower().strip()
        if act == "sleep":
            rc, _, err = _run_osascript('tell application "System Events" to sleep')
        elif act == "restart":
            rc, _, err = _run_osascript('tell application "System Events" to restart')
        elif act == "shutdown":
            rc, _, err = _run_osascript('tell application "System Events" to shut down')
        else:
            return ToolResult(ok=False, tool_name="system_power", message=f"Unknown power action: {action}")
        return ToolResult(
            ok=(rc == 0),
            tool_name="system_power",
            message=f"System {act} initiated." if rc == 0 else err,
        )

    def send_email(self, recipient: str, subject: str, body: str) -> ToolResult:
        script = f'''
tell application "Mail"
    set newMessage to make new outgoing message with properties {{subject:"{_escape_applescript(subject)}", content:"{_escape_applescript(body)}", visible:false}}
    tell newMessage
        make new to recipient at end of to recipients with properties {{address:"{_escape_applescript(recipient)}"}}
        send
    end tell
end tell
'''
        rc, _, err = _run_osascript(script, timeout=15.0)
        return ToolResult(
            ok=(rc == 0),
            tool_name="send_email",
            message=f"Email sent to {recipient}." if rc == 0 else f"Failed to send email: {err}",
        )

    def send_message(self, recipient: str, body: str) -> ToolResult:
        script = f'''
tell application "Messages"
    set targetService to 1st service whose service type = iMessage
    set targetBuddy to buddy "{_escape_applescript(recipient)}" of targetService
    send "{_escape_applescript(body)}" to targetBuddy
end tell
'''
        rc, _, err = _run_osascript(script, timeout=15.0)
        return ToolResult(
            ok=(rc == 0),
            tool_name="send_message",
            message=f"Message sent to {recipient}." if rc == 0 else f"Failed to send message: {err}",
        )

    # ------------------------------------------------------------------
    # 7. WhatsApp Desktop Automation
    # ------------------------------------------------------------------
    def whatsapp_send_message(self, recipient: str, message: str) -> ToolResult:
        from whatsapp_bridge import whatsapp_send_message as _wa_send

        res = _wa_send(recipient=recipient, message=message)
        return ToolResult(
            ok=bool(res.get("ok")),
            tool_name="whatsapp_send_message",
            message=str(res.get("message", "")),
            data=res,
        )

    def whatsapp_call(self, recipient: str, call_type: str = "voice") -> ToolResult:
        from whatsapp_bridge import whatsapp_call as _wa_call

        res = _wa_call(recipient=recipient, call_type=call_type)
        return ToolResult(
            ok=bool(res.get("ok")),
            tool_name="whatsapp_call",
            message=str(res.get("message", "")),
            data=res,
        )

    def whatsapp_answer_call(self) -> ToolResult:
        from whatsapp_bridge import whatsapp_answer_call as _wa_ans

        res = _wa_ans()
        return ToolResult(
            ok=bool(res.get("ok")),
            tool_name="whatsapp_answer_call",
            message=str(res.get("message", "")),
            data=res,
        )

    def whatsapp_decline_call(self) -> ToolResult:
        from whatsapp_bridge import whatsapp_decline_call as _wa_dec

        res = _wa_dec()
        return ToolResult(
            ok=bool(res.get("ok")),
            tool_name="whatsapp_decline_call",
            message=str(res.get("message", "")),
            data=res,
        )
