"""Safety Guardrails & Audit Logging for Project Jarvis (PRD Section 6).

Enforces spoken confirmation ("Yes, execute") prior to destructive actions:
- File deletion (`rm`, `unlink`, `shred`, `rmdir`, `delete_files`, `trash`)
- System shutdown / reboot / sleep (`shutdown`, `reboot`, `halt`, `pmset`)
- Sending emails or messages (`Mail`, `Messages`, `send_email`, `send_message`)
"""

from __future__ import annotations

import json
import logging
import re
import shlex
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from config.settings import JarvisConfig, load_config

logger = logging.getLogger(__name__)

# Patterns indicating potentially destructive shell or AppleScript commands
_DESTRUCTIVE_SHELL_RE = re.compile(
    r"(?:^|[;&|]\s*|\s)(?:rm|unlink|shred|srm|rmdir|shutdown|reboot|halt|poweroff|mkfs|dd)\b"
    r"|\bkillall\b|\bpkill\b|\bdiskutil\s+erase|\btrash\b|\bmv\s+.*\s+/dev/null",
    re.IGNORECASE,
)

_MESSAGING_APPLESCRIPT_RE = re.compile(
    r"\b(?:tell\s+application\s+\"(?:Mail|Messages)\".*\bsend\b|\bsend\s+outgoing\s+message\b)",
    re.IGNORECASE | re.DOTALL,
)

_SYSTEM_POWER_APPLESCRIPT_RE = re.compile(
    r"\b(?:shut\s+down|restart|sleep)\b",
    re.IGNORECASE,
)

_DESTRUCTIVE_TOOLS = {
    "delete_files",
    "system_power",
    "send_email",
    "send_message",
}


@dataclass(frozen=True)
class SafetyAssessment:
    requires_confirmation: bool
    category: str
    confirmation_prompt: str


def assess_action_safety(tool_name: str, args: dict[str, Any]) -> SafetyAssessment:
    """Determine if a tool invocation is destructive and craft a spoken confirmation prompt."""
    if tool_name == "delete_files":
        paths = args.get("paths") or ([args["path"]] if args.get("path") else [])
        count = len(paths)
        target_desc = f"{count} items" if count > 1 else (str(paths[0]) if paths else "selected files")
        return SafetyAssessment(
            requires_confirmation=True,
            category="file_deletion",
            confirmation_prompt=f"Confirm: delete {target_desc}? Say yes, execute to proceed.",
        )

    if tool_name == "system_power":
        action = str(args.get("action", "shutdown")).lower()
        return SafetyAssessment(
            requires_confirmation=True,
            category="system_power",
            confirmation_prompt=f"Confirm: {action} your Mac now? Say yes, execute to proceed.",
        )

    if tool_name in {"send_email", "send_message"}:
        recipient = args.get("recipient") or args.get("to") or "recipient"
        kind = "email" if tool_name == "send_email" else "message"
        return SafetyAssessment(
            requires_confirmation=True,
            category="outbound_message",
            confirmation_prompt=f"Confirm: send {kind} to {recipient}? Say yes, execute to proceed.",
        )

    if tool_name == "run_shell_command":
        cmd = str(args.get("command", ""))
        if _DESTRUCTIVE_SHELL_RE.search(cmd):
            # Build human-friendly summary for deletion or shutdown
            summary = _summarize_shell_command(cmd)
            return SafetyAssessment(
                requires_confirmation=True,
                category="destructive_shell",
                confirmation_prompt=f"Confirm: {summary}? Say yes, execute to proceed.",
            )

    if tool_name == "run_applescript":
        script = str(args.get("script", ""))
        if _MESSAGING_APPLESCRIPT_RE.search(script):
            return SafetyAssessment(
                requires_confirmation=True,
                category="outbound_message",
                confirmation_prompt="Confirm: send email or message via AppleScript? Say yes, execute to proceed.",
            )
        if _SYSTEM_POWER_APPLESCRIPT_RE.search(script):
            return SafetyAssessment(
                requires_confirmation=True,
                category="system_power",
                confirmation_prompt="Confirm: perform system power action? Say yes, execute to proceed.",
            )

    if tool_name in _DESTRUCTIVE_TOOLS:
        return SafetyAssessment(
            requires_confirmation=True,
            category="destructive_tool",
            confirmation_prompt=f"Confirm: execute {tool_name}? Say yes, execute to proceed.",
        )

    return SafetyAssessment(requires_confirmation=False, category="safe", confirmation_prompt="")


def _summarize_shell_command(cmd: str) -> str:
    lowered = cmd.lower()
    if any(tok in lowered for tok in ("shutdown", "reboot", "halt", "poweroff")):
        return "shut down or restart the system"
    if "rm " in lowered or lowered.startswith("rm"):
        try:
            tokens = shlex.split(cmd)
            targets = [t for t in tokens[1:] if not t.startswith("-")]
            if targets:
                return f"delete {', '.join(targets[:3])}"
        except ValueError:
            pass
        return f"run delete command {cmd[:60]}"
    return f"execute {cmd[:60]}"


def is_explicit_confirmation(transcript: str, required_phrase: str = "yes, execute") -> bool:
    """Verify the user spoke the explicit confirmation phrase (tolerant of punctuation)."""
    norm_text = re.sub(r"[^\w\s]", "", (transcript or "").lower()).strip()
    norm_req = re.sub(r"[^\w\s]", "", required_phrase.lower()).strip()
    if not norm_text or not norm_req:
        return False
    # Exact match or contains exact consecutive words "yes execute"
    return bool(re.search(rf"\b{re.escape(norm_req)}\b", norm_text))


class SafetyGuard:
    """Coordinates spoken confirmation prompts and security audit logging."""

    def __init__(
        self,
        config: Optional[JarvisConfig] = None,
        speak_fn: Optional[Callable[[str], None]] = None,
        listen_confirmation_fn: Optional[Callable[[float], Optional[str]]] = None,
    ) -> None:
        self.config = config or load_config()
        self.speak_fn = speak_fn
        self.listen_confirmation_fn = listen_confirmation_fn
        self.audit_path: Path = self.config.resolve_path(self.config.safety.audit_log_path)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, event_type: str, details: dict[str, Any]) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            **details,
        }
        try:
            with self.audit_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.debug("Failed to write audit log: %s", exc)

    def authorize(self, tool_name: str, args: dict[str, Any]) -> tuple[bool, str]:
        """Authorize a tool call; prompt & require biometrically verified 'Yes, execute' if destructive."""
        assessment = assess_action_safety(tool_name, args)
        if not assessment.requires_confirmation or not self.config.safety.confirm_destructive_actions:
            self.log_event("tool_authorized", {"tool": tool_name, "category": assessment.category})
            return True, "authorized"

        logger.warning(
            "Destructive action intercepted (%s: %s). Requesting voice confirmation.",
            assessment.category,
            tool_name,
        )
        if self.speak_fn is None or self.listen_confirmation_fn is None:
            self.log_event(
                "tool_denied_no_confirmation_channel",
                {"tool": tool_name, "category": assessment.category},
            )
            return False, "Confirmation required but voice confirmation channel is unavailable."

        self.speak_fn(assessment.confirmation_prompt)
        reply = self.listen_confirmation_fn(self.config.safety.confirmation_timeout_seconds)
        confirmed = is_explicit_confirmation(
            reply or "",
            required_phrase=self.config.safety.confirmation_phrase,
        )
        self.log_event(
            "destructive_confirmation",
            {
                "tool": tool_name,
                "category": assessment.category,
                "reply": reply,
                "confirmed": confirmed,
            },
        )
        if confirmed:
            return True, "confirmed_by_owner"
        return False, "Action cancelled: explicit 'Yes, execute' confirmation was not received."
