"""WhatsApp Desktop Automation & Incoming Call Monitor for Project Jarvis."""

from __future__ import annotations

import json
import logging
import re
import subprocess
import threading
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
CONTACTS_PATH = PROJECT_ROOT / "contacts.json"


@dataclass(frozen=True)
class ContactMatch:
    name: str
    phone_display: str
    phone_digits: str


def _escape_applescript(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _run_osascript(script: str, timeout: float = 8.0) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def normalize_phone_digits(phone: str) -> str:
    """Return digits-only phone string with country code for whatsapp:// URLs."""
    return re.sub(r"\D", "", (phone or "").strip())


class ContactBook:
    """Loads, updates, and resolves contact names/nicknames from contacts.json."""

    def __init__(self, path: Path = CONTACTS_PATH) -> None:
        self.path = path

    def load(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k).strip(): str(v).strip() for k, v in data.items() if str(k).strip()}
        except Exception as exc:
            logger.warning("Failed to load contacts.json: %s", exc)
        return {}

    def save(self, contacts: dict[str, str]) -> dict[str, str]:
        cleaned = {
            str(k).strip(): str(v).strip()
            for k, v in contacts.items()
            if str(k).strip() and str(v).strip()
        }
        self.path.write_text(json.dumps(cleaned, indent=2) + "\n", encoding="utf-8")
        return cleaned

    def upsert(self, name: str, phone: str) -> dict[str, str]:
        contacts = self.load()
        contacts[name.strip()] = phone.strip()
        return self.save(contacts)

    def delete(self, name: str) -> dict[str, str]:
        contacts = self.load()
        target = name.strip().lower()
        updated = {k: v for k, v in contacts.items() if k.lower() != target}
        return self.save(updated)

    def resolve(self, query: str) -> Optional[ContactMatch]:
        q = (query or "").strip()
        if not q:
            return None

        # Direct phone number input
        digits = normalize_phone_digits(q)
        if len(digits) >= 7 and sum(c.isdigit() for c in q) >= 7:
            return ContactMatch(name=q, phone_display=q, phone_digits=digits)

        contacts = self.load()
        q_lower = q.lower()
        # 1. Exact case-insensitive match
        for name, phone in contacts.items():
            if name.lower() == q_lower:
                return ContactMatch(
                    name=name,
                    phone_display=phone,
                    phone_digits=normalize_phone_digits(phone),
                )
        # 2. Substring match
        for name, phone in contacts.items():
            if q_lower in name.lower() or name.lower() in q_lower:
                return ContactMatch(
                    name=name,
                    phone_display=phone,
                    phone_digits=normalize_phone_digits(phone),
                )
        return None


def _open_whatsapp_chat_by_search(contact_name: str) -> bool:
    """Activate WhatsApp Desktop, Cmd+F search contact_name, Down + Return."""
    esc_name = _escape_applescript(contact_name)
    script = f'''
tell application "WhatsApp" to activate
delay 0.7
tell application "System Events"
    tell process "WhatsApp"
        set frontmost to true
        keystroke "f" using {{command down}}
        delay 0.35
        keystroke "a" using {{command down}}
        keystroke "{esc_name}"
        delay 0.85
        key code 125
        delay 0.15
        key code 36
    end tell
end tell
'''
    rc, _, err = _run_osascript(script, timeout=10.0)
    if rc != 0:
        logger.warning("WhatsApp search navigation error: %s", err)
    return rc == 0


def _click_ax_button_in_process(
    process_name: str,
    keywords: tuple[str, ...],
    max_nodes: int = 350,
) -> bool:
    """Walk AXUIElement tree of `process_name` and press first button matching `keywords`."""
    try:
        from AppKit import NSWorkspace
        from ApplicationServices import (
            AXUIElementCopyAttributeValue,
            AXUIElementCreateApplication,
            AXUIElementPerformAction,
            kAXChildrenAttribute,
            kAXDescriptionAttribute,
            kAXHelpAttribute,
            kAXPressAction,
            kAXRoleAttribute,
            kAXTitleAttribute,
        )

        pid = None
        for app in NSWorkspace.sharedWorkspace().runningApplications():
            if (app.localizedName() or "").lower() == process_name.lower():
                pid = app.processIdentifier()
                break
        if pid is None:
            return False

        app_ref = AXUIElementCreateApplication(pid)
        queue = [(app_ref, 0)]
        visited = 0
        while queue and visited < max_nodes:
            node, depth = queue.pop(0)
            visited += 1
            _, role = AXUIElementCopyAttributeValue(node, kAXRoleAttribute, None)
            role_str = str(role or "")
            texts: list[str] = []
            for attr in (kAXTitleAttribute, kAXDescriptionAttribute, kAXHelpAttribute):
                err, val = AXUIElementCopyAttributeValue(node, attr, None)
                if err == 0 and isinstance(val, str) and val.strip():
                    texts.append(val.strip().lower())

            combined = " ".join(texts)
            if combined and any(kw in combined for kw in keywords):
                if "Button" in role_str or not role_str:
                    if AXUIElementPerformAction(node, kAXPressAction) == 0:
                        return True
                # Fallback: try pressing even if role is AXGroup/MenuButton
                if AXUIElementPerformAction(node, kAXPressAction) == 0:
                    return True

            if depth < 8:
                c_err, children = AXUIElementCopyAttributeValue(node, kAXChildrenAttribute, None)
                if c_err == 0 and children:
                    for ch in list(children)[:50]:
                        queue.append((ch, depth + 1))
    except Exception as exc:
        logger.debug("AXUIElement button search failed in %s: %s", process_name, exc)
    return False


def _click_whatsapp_call_header_fallback(video: bool) -> bool:
    """Coordinate fallback relative to WhatsApp window top-right chat header."""
    try:
        import pyautogui

        script = '''
tell application "System Events"
    tell process "WhatsApp"
        set frontmost to true
        set winPos to position of front window
        set winSize to size of front window
        return (item 1 of winPos as text) & "," & (item 2 of winPos as text) & "," & (item 1 of winSize as text) & "," & (item 2 of winSize as text)
    end tell
end tell
'''
        rc, out, _ = _run_osascript(script)
        if rc != 0 or "," not in out:
            return False
        wx, wy, ww, _ = [int(float(v.strip())) for v in out.split(",")[:4]]
        # In WhatsApp Desktop macOS, Video call & Voice call icons sit near top-right header
        target_x = wx + ww - (112 if video else 72)
        target_y = wy + 52
        pyautogui.click(target_x, target_y)
        return True
    except Exception as exc:
        logger.debug("WhatsApp coordinate call fallback failed: %s", exc)
        return False


def whatsapp_send_message(
    recipient: str,
    message: str,
    contacts_path: Path = CONTACTS_PATH,
) -> dict[str, Any]:
    """Send a WhatsApp Desktop message via phone URL scheme or Cmd+F contact search."""
    book = ContactBook(contacts_path)
    match = book.resolve(recipient)
    display_name = match.name if match else recipient.strip()

    if match and match.phone_digits:
        encoded = urllib.parse.quote(message)
        url = f"whatsapp://send?phone={match.phone_digits}&text={encoded}"
        proc = subprocess.run(["open", url], capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            return {"ok": False, "message": f"Failed to open WhatsApp URL: {proc.stderr.strip()}"}

        time.sleep(1.5)
        send_script = '''
tell application "WhatsApp" to activate
delay 0.2
tell application "System Events"
    tell process "WhatsApp"
        set frontmost to true
        key code 36
    end tell
end tell
'''
        rc, _, err = _run_osascript(send_script, timeout=8.0)
        if rc != 0:
            return {"ok": False, "message": f"Opened chat for {display_name}, but failed to send: {err}"}
        return {
            "ok": True,
            "recipient": display_name,
            "method": "phone_url",
            "message": f"Message sent to {display_name}, sir.",
        }

    # Fallback: Cmd+F search inside WhatsApp Desktop
    if not _open_whatsapp_chat_by_search(display_name):
        return {"ok": False, "message": f"Could not locate {display_name} in WhatsApp Desktop."}

    time.sleep(0.45)
    esc_msg = _escape_applescript(message)
    type_script = f'''
tell application "System Events"
    tell process "WhatsApp"
        set frontmost to true
        keystroke "{esc_msg}"
        delay 0.15
        key code 36
    end tell
end tell
'''
    rc, _, err = _run_osascript(type_script, timeout=8.0)
    if rc != 0:
        return {"ok": False, "message": f"Selected {display_name}, but typing failed: {err}"}
    return {
        "ok": True,
        "recipient": display_name,
        "method": "search_bar",
        "message": f"Message sent to {display_name}, sir.",
    }


def whatsapp_call(
    recipient: str,
    call_type: str = "voice",
    contacts_path: Path = CONTACTS_PATH,
) -> dict[str, Any]:
    """Place a WhatsApp voice or video call and signal Jarvis to pause mic capture."""
    is_video = call_type.strip().lower() == "video"
    book = ContactBook(contacts_path)
    match = book.resolve(recipient)
    display_name = match.name if match else recipient.strip()

    if match and match.phone_digits:
        url = f"whatsapp://send?phone={match.phone_digits}"
        subprocess.run(["open", url], capture_output=True, text=True, check=False)
        time.sleep(1.4)
    else:
        if not _open_whatsapp_chat_by_search(display_name):
            return {"ok": False, "message": f"Could not open chat for {display_name} in WhatsApp."}
        time.sleep(0.5)

    _run_osascript('tell application "WhatsApp" to activate')
    time.sleep(0.3)

    keywords = (
        ("video call", "video") if is_video else ("voice call", "audio call", "call")
    )
    clicked = _click_ax_button_in_process("WhatsApp", keywords)
    if not clicked:
        clicked = _click_whatsapp_call_header_fallback(video=is_video)

    if not clicked:
        return {
            "ok": False,
            "message": f"Opened chat for {display_name}, but could not press the {call_type} call button.",
        }

    kind_label = "video calling" if is_video else "calling"
    return {
        "ok": True,
        "recipient": display_name,
        "call_type": "video" if is_video else "voice",
        "pause_jarvis_mic": True,
        "message": f"{kind_label.capitalize()} {display_name} on WhatsApp, sir. Pausing my microphone for your call.",
    }


def whatsapp_answer_call() -> dict[str, Any]:
    """Answer an incoming WhatsApp Desktop voice/video call and pause Jarvis mic."""
    _run_osascript('tell application "WhatsApp" to activate')
    time.sleep(0.2)
    clicked = _click_ax_button_in_process("WhatsApp", ("accept", "answer", "join"))
    if not clicked:
        clicked = _click_ax_button_in_process("NotificationCenter", ("accept", "answer"))
    if not clicked:
        return {"ok": False, "message": "No active incoming WhatsApp call Accept button found."}
    return {
        "ok": True,
        "pause_jarvis_mic": True,
        "message": "Call answered, sir. Pausing my microphone.",
    }


def whatsapp_decline_call() -> dict[str, Any]:
    """Decline an incoming WhatsApp Desktop voice/video call."""
    clicked = _click_ax_button_in_process("WhatsApp", ("decline", "reject", "end", "dismiss"))
    if not clicked:
        clicked = _click_ax_button_in_process("NotificationCenter", ("decline", "reject"))
    if not clicked:
        return {"ok": False, "message": "No active incoming WhatsApp call Decline button found."}
    return {
        "ok": True,
        "pause_jarvis_mic": False,
        "message": "Incoming WhatsApp call declined, sir.",
    }


def detect_incoming_whatsapp_call() -> Optional[dict[str, str]]:
    """Inspect WhatsApp windows & NotificationCenter for incoming voice/video calls."""
    script = '''
tell application "System Events"
    set outText to ""
    if exists process "WhatsApp" then
        tell process "WhatsApp"
            repeat with w in windows
                try
                    set wName to name of w as text
                    if wName contains "Incoming" or wName contains "voice call" or wName contains "video call" then
                        return "WHATSAPP|" & wName
                    end if
                    repeat with el in UI elements of w
                        try
                            set desc to (value of attribute "AXDescription" of el) as text
                            if desc contains "Incoming voice call" or desc contains "Incoming video call" then
                                return "WHATSAPP|" & desc
                            end if
                        end try
                        try
                            set ttl to (name of el) as text
                            if ttl contains "Incoming voice call" or ttl contains "Incoming video call" then
                                return "WHATSAPP|" & ttl
                            end if
                        end try
                    end repeat
                end try
            end repeat
        end tell
    end if
    if exists process "NotificationCenter" then
        tell process "NotificationCenter"
            repeat with w in windows
                try
                    repeat with el in entire contents of w
                        try
                            set val to (value of el) as text
                            if val contains "Incoming voice call" or val contains "Incoming video call" then
                                return "NOTIF|" & val
                            end if
                        end try
                    end repeat
                end try
            end repeat
        end tell
    end if
end tell
return ""
'''
    rc, out, _ = _run_osascript(script, timeout=3.0)
    if rc != 0 or not out:
        return None
    parts = out.split("|", 1)
    raw_label = parts[1].strip() if len(parts) > 1 else out
    call_type = "video" if "video" in raw_label.lower() else "voice"
    caller = re.sub(
        r"(?i)incoming\s+(?:whatsapp\s+)?(?:voice|video)?\s*call(?:\s+from)?",
        "",
        raw_label,
    ).strip(" :-|")
    return {
        "caller": caller or "Unknown caller",
        "call_type": call_type,
        "raw": raw_label,
    }


class WhatsAppCallMonitor:
    """Lightweight background monitor detecting incoming WhatsApp calls while Jarvis runs."""

    def __init__(
        self,
        on_incoming_call: Optional[Callable[[dict[str, str]], None]] = None,
        poll_interval: float = 2.0,
    ) -> None:
        self.on_incoming_call = on_incoming_call
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_announced_key: str = ""
        self._last_announced_ts: float = 0.0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="jarvis-whatsapp-call-monitor",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                info = detect_incoming_whatsapp_call()
                if info:
                    key = f"{info['caller']}:{info['call_type']}"
                    now = time.monotonic()
                    if key != self._last_announced_key or (now - self._last_announced_ts) > 25.0:
                        self._last_announced_key = key
                        self._last_announced_ts = now
                        if self.on_incoming_call is not None:
                            self.on_incoming_call(info)
                else:
                    # Reset once call banner clears
                    if time.monotonic() - self._last_announced_ts > 5.0:
                        self._last_announced_key = ""
            except Exception as exc:
                logger.debug("WhatsApp call monitor poll warning: %s", exc)
            self._stop_event.wait(timeout=self.poll_interval)
