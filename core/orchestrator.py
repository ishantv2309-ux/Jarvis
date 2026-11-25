"""Agent Brain & Tool Routing Orchestrator (PRD Phase 3).

Routes biometrically verified & transcribed user commands to `MacToolRegistry` via:
1. Ultra-low-latency deterministic fast-path router (sub-1.5s OS & media commands)
2. Gemini function calling (`google-genai`)
3. Local Ollama tool calling (`/api/chat`)
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from config.settings import JarvisConfig, load_config
from tools.mac_tools import TOOL_DECLARATIONS, MacToolRegistry, ToolResult

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are Jarvis, a concise, voice-only macOS system assistant operating exclusively "
    "for your authenticated owner. Execute requested actions using the provided tools, "
    "and respond in 1-2 short, natural spoken sentences without markdown or bullet lists."
)


@dataclass(frozen=True)
class OrchestratorResponse:
    spoken_text: str
    tool_results: list[ToolResult] = field(default_factory=list)
    route: str = "unknown"


def strip_wake_prefix(text: str) -> str:
    """Remove leading 'Jarvis,' / 'Hey Jarvis,' wake phrase from transcript."""
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^(?:hey\s+|ok\s+)?jarvis[\s,.:!?-]+", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def fast_path_route(text: str) -> list[tuple[str, dict[str, Any]]] | None:
    """Deterministic instant intent parser for common macOS & YouTube commands."""
    cmd = strip_wake_prefix(text)
    if not cmd:
        return None
    lower = cmd.lower().rstrip(".!?")

    # 1. YouTube direct search & play (FR-3)
    yt_match = re.match(
        r"^(?:search\s+(?:for\s+)?(?:and\s+play\s+)?|play\s+)(.+?)\s+on\s+youtube$",
        lower,
    )
    if yt_match:
        query = yt_match.group(1).strip()
        if query:
            return [("youtube_search_and_play", {"query": query})]

    yt_match2 = re.match(r"^search\s+youtube\s+(?:for\s+)?(.+)$", lower)
    if yt_match2:
        return [("youtube_search_and_play", {"query": yt_match2.group(1).strip()})]

    # 2. Media playback controls (FR-3)
    media_patterns = {
        r"^(?:pause|pause\s+(?:the\s+)?(?:video|playback|youtube|media))$": "pause",
        r"^(?:resume|play|resume\s+(?:the\s+)?(?:video|playback|youtube))$": "resume",
        r"^skip\s+(?:the\s+)?ad$": "skip_ad",
        r"^(?:toggle\s+)?fullscreen(?:\s+toggle)?$": "fullscreen",
        r"^next\s+(?:video|track)$": "next_video",
        r"^previous\s+(?:video|track)$": "previous_video",
        r"^mute\s+(?:the\s+)?(?:video|youtube|tab)$": "mute",
        r"^unmute\s+(?:the\s+)?(?:video|youtube|tab)$": "unmute",
    }
    for pat, act in media_patterns.items():
        if re.match(pat, lower):
            return [("media_control", {"action": act})]

    # 3. Application lifecycle (FR-2)
    if lower in ("what apps are running", "list running apps", "running applications"):
        return [("manage_application", {"action": "list_running"})]

    app_match = re.match(
        r"^(open|launch|start|focus|switch\s+to|minimize|hide|force[\s-]+quit|quit|close)\s+(.+)$",
        lower,
    )
    if app_match:
        verb, target = app_match.group(1), app_match.group(2).strip()
        # Avoid intercepting file/folder/delete requests or YouTube queries
        if not any(kw in target for kw in ("file", "folder", "directory", "on youtube")):
            action_map = {
                "open": "open",
                "launch": "open",
                "start": "open",
                "focus": "focus",
                "switch to": "focus",
                "minimize": "minimize",
                "hide": "hide",
                "quit": "quit",
                "close": "quit",
                "force quit": "force_quit",
                "force-quit": "force_quit",
            }
            norm_verb = re.sub(r"\s+", " ", verb)
            if norm_verb in action_map:
                return [("manage_application", {"action": action_map[norm_verb], "app_name": target})]

    # 4. Hardware & OS Settings (FR-2)
    vol_pct = re.match(r"^(?:set\s+)?(?:system\s+)?volume\s+(?:to\s+)?(\d+)\s*%?$", lower)
    if vol_pct:
        return [("control_os_setting", {"setting": "volume", "value": vol_pct.group(1)})]
    if lower in ("mute volume", "mute audio", "mute system", "mute"):
        return [("control_os_setting", {"setting": "volume", "value": "mute"})]
    if lower in ("unmute volume", "unmute audio", "unmute system", "unmute"):
        return [("control_os_setting", {"setting": "volume", "value": "unmute"})]
    if lower in ("volume up", "increase volume", "turn up volume", "louder"):
        return [("control_os_setting", {"setting": "volume", "value": "up"})]
    if lower in ("volume down", "decrease volume", "turn down volume", "quieter"):
        return [("control_os_setting", {"setting": "volume", "value": "down"})]

    if m := re.match(r"^(?:turn|switch)\s+(on|off)\s+dark\s+mode$", lower):
        return [("control_os_setting", {"setting": "dark_mode", "value": m.group(1)})]
    if lower in ("toggle dark mode", "enable dark mode", "disable dark mode"):
        val = "on" if "enable" in lower else ("off" if "disable" in lower else "toggle")
        return [("control_os_setting", {"setting": "dark_mode", "value": val})]

    if m := re.match(r"^(?:turn|switch|enable|disable)\s+(on|off)?\s*wi-?fi\s*(on|off)?$", lower):
        val = m.group(1) or m.group(2) or ("on" if "enable" in lower else "off")
        return [("control_os_setting", {"setting": "wifi", "value": val})]

    if m := re.match(r"^(?:turn|switch|enable|disable)\s+(on|off)?\s*bluetooth\s*(on|off)?$", lower):
        val = m.group(1) or m.group(2) or ("on" if "enable" in lower else "off")
        return [("control_os_setting", {"setting": "bluetooth", "value": val})]

    if m := re.match(r"^(?:turn|switch|enable|disable)\s+(on|off)?\s*do\s+not\s+disturb\s*(on|off)?$", lower):
        val = m.group(1) or m.group(2) or ("on" if "enable" in lower else "off")
        return [("control_os_setting", {"setting": "do_not_disturb", "value": val})]

    if lower in ("brightness up", "increase brightness", "brighter"):
        return [("control_os_setting", {"setting": "brightness", "value": "up"})]
    if lower in ("brightness down", "decrease brightness", "dimmer"):
        return [("control_os_setting", {"setting": "brightness", "value": "down"})]

    # 5. Screen awareness & Spotlight search
    if lower in ("what window is active", "active window", "what app is active", "what is on screen"):
        return [("get_active_window", {})]

    if m := re.match(r"^(?:find|search\s+(?:files\s+)?for|spotlight)\s+(.+)$", lower):
        return [("search_files", {"query": m.group(1).strip()})]

    # 5b. WhatsApp Desktop automation (send message, voice/video call, answer/decline)
    raw_cmd = cmd.strip().rstrip(".!?")
    if re.match(r"^(?:answer|pick\s*up)(?:\s+(?:the\s+)?(?:whatsapp\s+)?call)?$", lower):
        return [("whatsapp_answer_call", {})]
    if re.match(r"^(?:decline|reject|ignore)(?:\s+(?:the\s+)?(?:whatsapp\s+)?call)?$", lower):
        return [("whatsapp_decline_call", {})]

    wa_msg = re.match(
        r"^(?:send\s+(?:a\s+)?whatsapp\s+(?:message\s+)?to|message)\s+(.+?)\s+(?:on\s+whatsapp\s+)?(?:saying|that)\s+(.+)$",
        raw_cmd,
        flags=re.IGNORECASE,
    )
    if wa_msg:
        return [("whatsapp_send_message", {"recipient": wa_msg.group(1).strip(), "message": wa_msg.group(2).strip()})]

    wa_msg2 = re.match(
        r"^whatsapp\s+(.+?)\s+(?:saying|that)\s+(.+)$",
        raw_cmd,
        flags=re.IGNORECASE,
    )
    if wa_msg2:
        return [("whatsapp_send_message", {"recipient": wa_msg2.group(1).strip(), "message": wa_msg2.group(2).strip()})]

    wa_call = re.match(
        r"^(?:(video|voice|audio)\s+)?call\s+(.+?)\s+on\s+whatsapp$",
        raw_cmd,
        flags=re.IGNORECASE,
    )
    if wa_call:
        ctype = "video" if (wa_call.group(1) or "").lower() == "video" else "voice"
        return [("whatsapp_call", {"recipient": wa_call.group(2).strip(), "call_type": ctype})]

    # 6. Destructive actions (routed safely through SafetyGuard)
    if m := re.match(r"^delete\s+(?:the\s+file\s+)?(.+)$", cmd.rstrip(".!?"), flags=re.IGNORECASE):
        return [("delete_files", {"paths": [m.group(1).strip()]})]

    if lower in ("shut down", "shutdown", "shut down mac", "turn off mac"):
        return [("system_power", {"action": "shutdown"})]
    if lower in ("restart mac", "reboot", "restart computer"):
        return [("system_power", {"action": "restart"})]

    return None


class AgentOrchestrator:
    """Routes verified transcripts via fast-path, Gemini, or Ollama to MacToolRegistry."""

    def __init__(
        self,
        config: Optional[JarvisConfig] = None,
        tools: Optional[MacToolRegistry] = None,
    ) -> None:
        self.config = config or load_config()
        self.tools = tools or MacToolRegistry()

    def handle_command(self, transcript: str) -> OrchestratorResponse:
        cleaned = strip_wake_prefix(transcript)
        if not cleaned:
            return OrchestratorResponse(spoken_text="", route="empty")

        # 1. Fast-path deterministic routing (sub-1.5s latency target)
        if self.config.llm.fast_path_enabled:
            calls = fast_path_route(cleaned)
            if calls:
                results = [self.tools.execute(name, args) for name, args in calls]
                spoken = " ".join(r.message for r in results if r.message).strip()
                return OrchestratorResponse(
                    spoken_text=spoken or "Done.",
                    tool_results=results,
                    route="fast_path",
                )

        # 2. LLM Orchestrator selection
        provider = self.config.llm.provider.lower()
        if provider in ("auto", "gemini") and self._has_gemini_key():
            try:
                return self._run_gemini(cleaned)
            except Exception as exc:
                logger.warning("Gemini orchestration failed (%s); trying fallback.", exc)
                if provider == "gemini":
                    return OrchestratorResponse(
                        spoken_text=f"Gemini request failed: {exc}",
                        route="gemini_error",
                    )

        if provider in ("auto", "ollama"):
            try:
                return self._run_ollama(cleaned)
            except Exception as exc:
                logger.warning("Ollama orchestration unavailable (%s).", exc)
                if provider == "ollama":
                    return OrchestratorResponse(
                        spoken_text="Local Ollama model is not reachable.",
                        route="ollama_error",
                    )

        return OrchestratorResponse(
            spoken_text="I could not match that command locally, and no LLM backend is configured.",
            route="unmatched",
        )

    def _has_gemini_key(self) -> bool:
        return bool(os.environ.get(self.config.llm.gemini_api_key_env, "").strip())

    def _run_gemini(self, prompt: str) -> OrchestratorResponse:
        from google import genai
        from google.genai import types

        api_key = os.environ[self.config.llm.gemini_api_key_env].strip()
        client = genai.Client(api_key=api_key)

        fn_decls = [
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=t["parameters"],
            )
            for t in TOOL_DECLARATIONS
        ]
        tool_obj = types.Tool(function_declarations=fn_decls)

        response = client.models.generate_content(
            model=self.config.llm.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                tools=[tool_obj],
                temperature=0.1,
            ),
        )

        results: list[ToolResult] = []
        fn_calls = getattr(response, "function_calls", None) or []
        if fn_calls:
            fn_parts = []
            for fc in fn_calls:
                args = dict(fc.args or {})
                res = self.tools.execute(fc.name, args)
                results.append(res)
                fn_parts.append(
                    types.Part.from_function_response(
                        name=fc.name,
                        response={"ok": res.ok, "message": res.message, "data": res.data},
                    )
                )

            followup = client.models.generate_content(
                model=self.config.llm.gemini_model,
                contents=[
                    types.Content(role="user", parts=[types.Part.from_text(text=prompt)]),
                    response.candidates[0].content,
                    types.Content(role="user", parts=fn_parts),
                ],
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    temperature=0.2,
                ),
            )
            spoken = (followup.text or "").strip() or (
                " ".join(r.message for r in results) if results else "Done."
            )
            return OrchestratorResponse(spoken_text=spoken, tool_results=results, route="gemini")

        spoken = (response.text or "Done.").strip()
        return OrchestratorResponse(spoken_text=spoken, tool_results=[], route="gemini")

    def _run_ollama(self, prompt: str) -> OrchestratorResponse:
        url = f"{self.config.llm.ollama_base_url.rstrip('/')}/api/chat"
        ollama_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in TOOL_DECLARATIONS
        ]
        payload = {
            "model": self.config.llm.ollama_model,
            "stream": False,
            "keep_alive": "10m",
            "options": {"temperature": 0.1},
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "tools": ollama_tools,
        }
        with httpx.Client(timeout=self.config.llm.ollama_timeout_seconds) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        msg = data.get("message") or {}
        tool_calls = msg.get("tool_calls") or []
        results: list[ToolResult] = []
        for tc in tool_calls:
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            raw_args = fn.get("arguments") or {}
            if isinstance(raw_args, str):
                raw_args = json.loads(raw_args)
            if name:
                results.append(self.tools.execute(name, raw_args))

        if results:
            spoken = " ".join(r.message for r in results if r.message).strip() or "Done."
        else:
            spoken = (msg.get("content") or "Done.").strip()
        return OrchestratorResponse(spoken_text=spoken, tool_results=results, route="ollama")
