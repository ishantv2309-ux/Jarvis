"""Multi-Step Agentic Tool-Calling & Conversational Engine (Web Jarvis Phase 3)."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import httpx

from config.settings import JarvisConfig, load_config
from core.orchestrator import fast_path_route, strip_wake_prefix
from core.safety import SafetyGuard, assess_action_safety, is_explicit_confirmation
from core.standby_listener import matches_sleep_phrase
from tools.mac_tools import TOOL_DECLARATIONS, MacToolRegistry, ToolResult

logger = logging.getLogger(__name__)

_WEB_SYSTEM_PROMPT = (
    "You are J.A.R.V.I.S., a refined, composed, dry-witted British AI butler operating "
    "your owner's macOS workstation. Address the user respectfully as 'sir'. "
    "You converse naturally while planning and executing multi-step macOS workflows "
    "via the provided tools. Confirm completed actions concisely in 1-3 spoken sentences "
    "without robotic filler, markdown formatting, or bullet points."
)


@dataclass
class PendingConfirmation:
    tool_name: str
    args: dict[str, Any]
    prompt: str
    remaining_clauses: list[str] = field(default_factory=list)


@dataclass
class AgentStepLog:
    step_index: int
    tool_name: str
    args: dict[str, Any]
    ok: bool
    message: str
    latency_ms: float
    pause_mic: bool = False


@dataclass
class AgentTurnResult:
    spoken_reply: str
    steps: list[AgentStepLog] = field(default_factory=list)
    awaiting_confirmation: bool = False
    confirmation_prompt: str = ""
    sleep_requested: bool = False
    pause_mic: bool = False
    route: str = "ollama"


StepCallback = Callable[[dict[str, Any]], Awaitable[None]]


def split_multi_step_clauses(text: str) -> list[str]:
    """Split compound fast-path commands like 'open VS Code, create folder X, and play Y on YouTube'."""
    cleaned = strip_wake_prefix(text)
    if not cleaned:
        return []
    parts = re.split(r"\s*(?:,\s*(?:and\s+)?|\bthen\b|\band\s+(?=(?:open|launch|close|quit|play|search|set|turn|pause|resume|mute|unmute|create|make|delete|shut)))", cleaned, flags=re.IGNORECASE)
    return [p.strip(" ,.") for p in parts if p and p.strip(" ,.")]


def fast_path_clause(clause: str) -> list[tuple[str, dict[str, Any]]] | None:
    """Extend core fast_path_route with folder creation on Desktop/Home."""
    direct = fast_path_route(clause)
    if direct:
        return direct

    raw_clause = clause.strip().rstrip(".!?")
    mkdir_match = re.match(
        r"^(?:create|make)\s+(?:a\s+)?(?:new\s+)?(?:project\s+)?folder\s+(?:named\s+|called\s+)?(.+?)(?:\s+(?:on|in)\s+(desktop|documents|downloads|home))?$",
        raw_clause,
        flags=re.IGNORECASE,
    )
    if mkdir_match:
        name = re.sub(r"[^\w\-. ]", "_", mkdir_match.group(1).strip())
        location = (mkdir_match.group(2) or "desktop").lower()
        base_map = {
            "desktop": Path.home() / "Desktop",
            "documents": Path.home() / "Documents",
            "downloads": Path.home() / "Downloads",
            "home": Path.home(),
        }
        target = str(base_map.get(location, Path.home() / "Desktop") / name)
        return [("manage_files", {"action": "mkdir", "source": target})]
    return None


def _is_web_confirmation(text: str) -> bool:
    """Accept 'Yes, confirm' (Web PRD) or 'Yes, execute' (Desktop PRD)."""
    return is_explicit_confirmation(text, "yes, confirm") or is_explicit_confirmation(
        text, "yes, execute"
    )


class WebAgentEngine:
    """Stateful conversational + multi-step macOS tool-calling engine for a web session."""

    def __init__(
        self,
        config: Optional[JarvisConfig] = None,
        tools: Optional[MacToolRegistry] = None,
    ) -> None:
        self.config = config or load_config()
        # Use non-blocking SafetyGuard without local PyAudio confirmation;
        # WebAgentEngine handles spoken confirmation directly over the web session.
        self.safety_guard = SafetyGuard(config=self.config)
        self.tools = tools or MacToolRegistry(safety_guard=self.safety_guard)
        self.history: list[dict[str, Any]] = []
        self.pending_confirmation: Optional[PendingConfirmation] = None

    def clear_history(self) -> None:
        self.history.clear()
        self.pending_confirmation = None

    async def handle_user_input(
        self,
        text: str,
        on_step: Optional[StepCallback] = None,
    ) -> AgentTurnResult:
        """Process one verified voice/text turn, supporting multi-step tools & conversation."""
        cleaned = strip_wake_prefix(text)
        if not cleaned:
            return AgentTurnResult(spoken_reply="", route="empty")

        if matches_sleep_phrase(cleaned):
            return AgentTurnResult(
                spoken_reply="Entering standby mode, sir. Just say Jarvis wake up when you need me.",
                sleep_requested=True,
                route="standby_command",
            )

        # 1. Handle active destructive confirmation prompt ("Yes, confirm" / "Yes, execute")
        if self.pending_confirmation is not None:
            pending = self.pending_confirmation
            self.pending_confirmation = None
            if _is_web_confirmation(cleaned):
                self.safety_guard.log_event(
                    "web_destructive_confirmed",
                    {"tool": pending.tool_name, "reply": cleaned},
                )
                handler = self.tools._handlers.get(pending.tool_name)
                if handler is None:
                    return AgentTurnResult(
                        spoken_reply=f"Unknown tool {pending.tool_name}.",
                        route="confirmation",
                    )
                if on_step:
                    await on_step({"status": "running", "tool": pending.tool_name, "args": pending.args})
                res: ToolResult = handler(**pending.args)
                step = AgentStepLog(
                    step_index=1,
                    tool_name=pending.tool_name,
                    args=pending.args,
                    ok=res.ok,
                    message=res.message,
                    latency_ms=res.latency_ms,
                )
                if on_step:
                    await on_step({"status": "done" if res.ok else "error", "tool": pending.tool_name, "message": res.message})
                return AgentTurnResult(
                    spoken_reply=res.message or "Confirmed and executed.",
                    steps=[step],
                    route="confirmation",
                )

            self.safety_guard.log_event(
                "web_destructive_cancelled",
                {"tool": pending.tool_name, "reply": cleaned},
            )
            return AgentTurnResult(
                spoken_reply="Cancelled. I did not execute that action.",
                route="confirmation_cancelled",
            )

        # 2. Check multi-clause deterministic fast-path (e.g. "Open VS Code, create folder X, and play Y on YouTube")
        if self.config.llm.fast_path_enabled:
            clauses = split_multi_step_clauses(cleaned)
            planned_calls: list[tuple[str, dict[str, Any]]] = []
            all_matched = bool(clauses)
            for clause in clauses:
                matched = fast_path_clause(clause)
                if matched:
                    planned_calls.extend(matched)
                else:
                    all_matched = False
                    break

            if all_matched and planned_calls:
                return await self._execute_planned_calls(
                    user_text=cleaned,
                    calls=planned_calls,
                    route="fast_path_multi" if len(planned_calls) > 1 else "fast_path",
                    on_step=on_step,
                )

        # 3. Conversational + Multi-Step Tool-Calling Loop via Ollama (or Gemini)
        provider = self.config.llm.provider.lower()
        if provider in ("auto", "gemini") and os.environ.get(
            self.config.llm.gemini_api_key_env, ""
        ).strip():
            try:
                return await self._run_gemini_turn(cleaned, on_step=on_step)
            except Exception as exc:
                logger.warning("Gemini failed (%s); falling back to Ollama.", exc)

        if provider in ("auto", "ollama"):
            try:
                return await self._run_ollama_loop(cleaned, on_step=on_step)
            except Exception as exc:
                logger.warning("Ollama loop failed: %s", exc)
                return AgentTurnResult(
                    spoken_reply="Local Ollama service is not responding right now. Please check that Ollama is running.",
                    route="ollama_error",
                )

        return AgentTurnResult(
            spoken_reply="I did not match a local command, and no LLM provider is enabled.",
            route="unmatched",
        )

    async def _execute_tool_with_confirmation_check(
        self,
        step_idx: int,
        tool_name: str,
        args: dict[str, Any],
        on_step: Optional[StepCallback],
    ) -> tuple[Optional[AgentStepLog], Optional[PendingConfirmation]]:
        assessment = assess_action_safety(tool_name, args)
        if assessment.requires_confirmation and self.config.safety.confirm_destructive_actions:
            prompt = assessment.confirmation_prompt.replace("yes, execute", "Yes, confirm")
            if on_step:
                await on_step({"status": "awaiting_confirmation", "tool": tool_name, "message": prompt})
            return None, PendingConfirmation(tool_name=tool_name, args=args, prompt=prompt)

        if on_step:
            await on_step({"status": "running", "tool": tool_name, "args": args, "step": step_idx})

        res = self.tools.execute(tool_name, args)
        pause_mic = bool((res.data or {}).get("pause_jarvis_mic"))
        step = AgentStepLog(
            step_index=step_idx,
            tool_name=tool_name,
            args=args,
            ok=res.ok,
            message=res.message,
            latency_ms=res.latency_ms,
            pause_mic=pause_mic,
        )
        if on_step:
            await on_step(
                {
                    "status": "done" if res.ok else "error",
                    "tool": tool_name,
                    "message": res.message,
                    "step": step_idx,
                }
            )
        return step, None

    async def _execute_planned_calls(
        self,
        user_text: str,
        calls: list[tuple[str, dict[str, Any]]],
        route: str,
        on_step: Optional[StepCallback],
    ) -> AgentTurnResult:
        steps: list[AgentStepLog] = []
        for idx, (tool_name, args) in enumerate(calls, start=1):
            step, pending = await self._execute_tool_with_confirmation_check(
                idx, tool_name, args, on_step
            )
            if pending is not None:
                self.pending_confirmation = pending
                return AgentTurnResult(
                    spoken_reply=pending.prompt,
                    steps=steps,
                    awaiting_confirmation=True,
                    confirmation_prompt=pending.prompt,
                    route=route,
                )
            if step is not None:
                steps.append(step)

        raw_summary = " ".join(s.message for s in steps if s.message).strip() or "Done"
        spoken = raw_summary.rstrip(".") + ", sir." if not raw_summary.lower().endswith("sir.") else raw_summary
        pause_mic = any(s.pause_mic for s in steps)
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": spoken})
        self.history = self.history[-16:]
        return AgentTurnResult(spoken_reply=spoken, steps=steps, pause_mic=pause_mic, route=route)

    async def _run_ollama_loop(
        self,
        prompt: str,
        on_step: Optional[StepCallback],
    ) -> AgentTurnResult:
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

        messages: list[dict[str, Any]] = [{"role": "system", "content": _WEB_SYSTEM_PROMPT}]
        messages.extend(self.history[-10:])
        messages.append({"role": "user", "content": prompt})

        steps: list[AgentStepLog] = []
        spoken_reply = ""

        async with httpx.AsyncClient(timeout=self.config.llm.ollama_timeout_seconds) as client:
            for round_idx in range(1, 6):
                payload = {
                    "model": self.config.llm.ollama_model,
                    "stream": False,
                    "keep_alive": "10m",
                    "options": {"temperature": 0.25},
                    "messages": messages,
                    "tools": ollama_tools,
                }
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                msg = data.get("message") or {}
                tool_calls = msg.get("tool_calls") or []

                if not tool_calls:
                    spoken_reply = (msg.get("content") or "").strip()
                    break

                messages.append(msg)
                for tc in tool_calls:
                    fn = tc.get("function") or {}
                    name = fn.get("name", "")
                    raw_args = fn.get("arguments") or {}
                    if isinstance(raw_args, str):
                        raw_args = json.loads(raw_args)
                    if not name:
                        continue

                    step, pending = await self._execute_tool_with_confirmation_check(
                        len(steps) + 1, name, raw_args, on_step
                    )
                    if pending is not None:
                        self.pending_confirmation = pending
                        return AgentTurnResult(
                            spoken_reply=pending.prompt,
                            steps=steps,
                            awaiting_confirmation=True,
                            confirmation_prompt=pending.prompt,
                            route="ollama",
                        )
                    if step is not None:
                        steps.append(step)
                        messages.append(
                            {
                                "role": "tool",
                                "content": json.dumps(
                                    {"ok": step.ok, "message": step.message}
                                ),
                            }
                        )

        if not spoken_reply:
            spoken_reply = (
                " ".join(s.message for s in steps if s.message).strip() or "Done."
            )

        self.history.append({"role": "user", "content": prompt})
        self.history.append({"role": "assistant", "content": spoken_reply})
        self.history = self.history[-16:]
        return AgentTurnResult(
            spoken_reply=spoken_reply,
            steps=steps,
            pause_mic=any(s.pause_mic for s in steps),
            route="ollama",
        )

    async def _run_gemini_turn(
        self,
        prompt: str,
        on_step: Optional[StepCallback],
    ) -> AgentTurnResult:
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
                system_instruction=_WEB_SYSTEM_PROMPT,
                tools=[tool_obj],
                temperature=0.2,
            ),
        )
        steps: list[AgentStepLog] = []
        fn_calls = getattr(response, "function_calls", None) or []
        if fn_calls:
            fn_parts = []
            for fc in fn_calls:
                args = dict(fc.args or {})
                step, pending = await self._execute_tool_with_confirmation_check(
                    len(steps) + 1, fc.name, args, on_step
                )
                if pending is not None:
                    self.pending_confirmation = pending
                    return AgentTurnResult(
                        spoken_reply=pending.prompt,
                        steps=steps,
                        awaiting_confirmation=True,
                        confirmation_prompt=pending.prompt,
                        route="gemini",
                    )
                if step is not None:
                    steps.append(step)
                    fn_parts.append(
                        types.Part.from_function_response(
                            name=fc.name,
                            response={"ok": step.ok, "message": step.message},
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
                    system_instruction=_WEB_SYSTEM_PROMPT, temperature=0.2
                ),
            )
            spoken = (followup.text or "").strip() or " ".join(s.message for s in steps)
            return AgentTurnResult(spoken_reply=spoken, steps=steps, route="gemini")

        spoken = (response.text or "Done.").strip()
        return AgentTurnResult(spoken_reply=spoken, steps=[], route="gemini")
