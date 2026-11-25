"""Global Emergency Kill-Switch Hotkey (PRD Section 6: Cmd + Shift + Escape).

Uses macOS Quartz CGEventTap (when Accessibility is granted) to intercept
the global shortcut and immediately terminate the Jarvis daemon and release
microphone access.
"""

from __future__ import annotations

import logging
import os
import signal
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# macOS virtual keycodes
_KEYCODES = {
    "escape": 53,
    "esc": 53,
    "k": 40,
    "j": 38,
}


def parse_hotkey(spec: str) -> tuple[set[str], int]:
    """Parse hotkey spec like 'cmd+shift+escape' into modifier names and virtual keycode."""
    parts = [p.strip().lower() for p in spec.split("+") if p.strip()]
    if not parts:
        return {"cmd", "shift"}, 53
    key = parts[-1]
    mods = set(parts[:-1])
    keycode = _KEYCODES.get(key, 53)
    return mods, keycode


class GlobalKillSwitch:
    """Listens globally for Cmd+Shift+Escape and invokes the emergency shutdown callback."""

    def __init__(
        self,
        hotkey: str = "cmd+shift+escape",
        on_trigger: Optional[Callable[[], None]] = None,
    ) -> None:
        self.hotkey = hotkey
        self.mods, self.keycode = parse_hotkey(hotkey)
        self.on_trigger = on_trigger
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._run_loop = None

    def start(self) -> bool:
        """Start the global CGEventTap listener in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            return True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_quartz_tap,
            daemon=True,
            name="jarvis-killswitch",
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()
        if self._run_loop is not None:
            try:
                import Quartz

                Quartz.CFRunLoopStop(self._run_loop)
            except Exception:
                pass

    def _trigger(self) -> None:
        logger.critical("EMERGENCY KILL-SWITCH (%s) TRIGGERED! Shutting down.", self.hotkey)
        if self.on_trigger is not None:
            try:
                self.on_trigger()
            except Exception as exc:
                logger.error("Kill-switch callback error: %s", exc)
        else:
            os.kill(os.getpid(), signal.SIGTERM)

    def _run_quartz_tap(self) -> None:
        try:
            import Quartz
        except Exception as exc:
            logger.warning("Quartz unavailable; global kill-switch disabled: %s", exc)
            return

        required_mask = 0
        if "cmd" in self.mods or "command" in self.mods:
            required_mask |= Quartz.kCGEventFlagMaskCommand
        if "shift" in self.mods:
            required_mask |= Quartz.kCGEventFlagMaskShift
        if "alt" in self.mods or "option" in self.mods:
            required_mask |= Quartz.kCGEventFlagMaskAlternate
        if "ctrl" in self.mods or "control" in self.mods:
            required_mask |= Quartz.kCGEventFlagMaskControl

        target_keycode = self.keycode

        def _callback(proxy, event_type, event, refcon):
            if event_type == Quartz.kCGEventKeyDown:
                code = Quartz.CGEventGetIntegerValueField(
                    event, Quartz.kCGKeyboardEventKeycode
                )
                flags = Quartz.CGEventGetFlags(event)
                if code == target_keycode and (flags & required_mask) == required_mask:
                    threading.Thread(
                        target=self._trigger, daemon=True, name="killswitch-exec"
                    ).start()
                    return None
            return event

        event_mask = Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown)
        tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionDefault,
            event_mask,
            _callback,
            None,
        )
        if tap is None:
            logger.warning(
                "Could not create CGEventTap for %s. Grant Accessibility permission in "
                "System Settings -> Privacy & Security -> Accessibility.",
                self.hotkey,
            )
            return

        source = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
        self._run_loop = Quartz.CFRunLoopGetCurrent()
        Quartz.CFRunLoopAddSource(self._run_loop, source, Quartz.kCFRunLoopCommonModes)
        Quartz.CGEventTapEnable(tap, True)
        logger.info("Global kill-switch active on %s.", self.hotkey)
        Quartz.CFRunLoopRun()
