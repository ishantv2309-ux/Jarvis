"""Standalone Web & Strict On-Demand Browser-Mic Bridge for Project Jarvis."""

from __future__ import annotations

import argparse
import asyncio
import base64
import logging
import os
import re
import signal
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent_tools import WebAgentEngine
from config.settings import JarvisConfig, load_config
from core.standby_listener import extract_jarvis_command
from scripts.check_permissions import (
    check_accessibility,
    check_automation_system_events,
    check_full_disk_access,
)

try:
    from tts_jarvis import CHIME_PATH, ensure_boot_chime_wav
except Exception:  # pragma: no cover - defensive fallback
    CHIME_PATH = Path(__file__).resolve().parent / "static" / "chime.wav"

    def ensure_boot_chime_wav(path: Path = CHIME_PATH) -> Path:
        return path

from voice_auth import WebAudioSegmenter, WebVoiceAuthService
from whatsapp_bridge import ContactBook, WhatsAppCallMonitor

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

FOLLOW_UP_WINDOW_SECONDS = 10.0

logger = logging.getLogger("jarvis.web")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


class AppState:
    def __init__(self, config: JarvisConfig) -> None:
        self.config = config
        self.voice_auth = WebVoiceAuthService(config=config)
        self.contacts = ContactBook()
        self.active_sockets: set[WebSocket] = set()
        self.auto_shutdown_on_close: bool = False
        self.shutdown_grace_seconds: float = 2.5
        self._shutdown_task: Optional[asyncio.Task] = None
        self._terminated: bool = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.whatsapp_monitor: Optional[WhatsAppCallMonitor] = None
        self.incoming_call_ts: float = 0.0
        ensure_boot_chime_wav()

    def cancel_pending_shutdown(self) -> None:
        if self._shutdown_task and not self._shutdown_task.done():
            self._shutdown_task.cancel()
            self._shutdown_task = None

    def schedule_shutdown_if_idle(self) -> None:
        if self._terminated or self.active_sockets or not self.auto_shutdown_on_close:
            return
        self.cancel_pending_shutdown()

        async def _delayed() -> None:
            try:
                await asyncio.sleep(self.shutdown_grace_seconds)
                if not self.active_sockets:
                    logger.info("Browser tab closed. Shutting down Jarvis web bridge.")
                    trigger_process_exit()
            except asyncio.CancelledError:
                pass

        self._shutdown_task = asyncio.create_task(_delayed())


def trigger_process_exit() -> None:
    try:
        os.kill(os.getpid(), signal.SIGTERM)
    except Exception:
        os._exit(0)


def kill_stale_port(port: int = 8000) -> None:
    """Kill any stale process currently bound to `port` before starting Uvicorn."""
    try:
        proc = subprocess.run(
            ["lsof", f"-ti:{port}"],
            capture_output=True,
            text=True,
            check=False,
        )
        pids = [p.strip() for p in proc.stdout.splitlines() if p.strip()]
        my_pid = str(os.getpid())
        for pid in pids:
            if pid != my_pid:
                subprocess.run(["kill", "-9", pid], check=False)
        if any(p != my_pid for p in pids):
            time.sleep(0.4)
    except Exception as exc:
        logger.debug("Port %d cleanup skipped: %s", port, exc)


class SettingsUpdatePayload(BaseModel):
    llm_provider: Optional[str] = None
    ollama_model: Optional[str] = None
    tts_engine: Optional[str] = None
    edge_voice: Optional[str] = None
    macos_voice: Optional[str] = None
    auto_shutdown_on_close: Optional[bool] = None
    confirm_destructive_actions: Optional[bool] = None


class ContactPayload(BaseModel):
    name: str
    phone: str


class BeaconPayload(BaseModel):
    reason: str = "tab_closed"
    navigating: bool = False


def create_app(
    config: Optional[JarvisConfig] = None,
    warmup_models: bool = True,
    initial_mode: str = "ACTIVE",
    enable_standby_listener: bool = False,
    enable_whatsapp_monitor: bool = True,
) -> FastAPI:
    cfg = config or load_config()
    state = AppState(cfg)

    async def _broadcast_incoming_call(info: dict[str, str]) -> None:
        caller = info.get("caller", "Unknown caller")
        call_type = info.get("call_type", "voice")
        announcement = f"Sir, incoming WhatsApp {call_type} call from {caller}. Shall I answer?"
        state.incoming_call_ts = time.monotonic()
        print(f'[WhatsApp] Incoming {call_type} call from "{caller}"', flush=True)

        if state.active_sockets:
            synth = await state.voice_auth.synthesize_tts_async(announcement)
            payload = {
                "type": "incoming_call",
                "caller": caller,
                "call_type": call_type,
                "text": announcement,
                "follow_up_seconds": FOLLOW_UP_WINDOW_SECONDS,
                "audio_b64": synth.audio_b64 if synth else None,
                "mime_type": synth.mime_type if synth else "audio/wav",
            }
            for ws in list(state.active_sockets):
                try:
                    await ws.send_json(payload)
                except Exception:
                    pass
        else:
            await asyncio.to_thread(state.voice_auth.tts.speak_local, announcement)

    def _on_incoming_whatsapp_call(info: dict[str, str]) -> None:
        if state.loop is not None and state.loop.is_running():
            asyncio.run_coroutine_threadsafe(_broadcast_incoming_call(info), state.loop)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state.loop = asyncio.get_running_loop()
        if warmup_models:
            asyncio.create_task(asyncio.to_thread(state.voice_auth.warmup))
        if enable_whatsapp_monitor:
            mon = WhatsAppCallMonitor(on_incoming_call=_on_incoming_whatsapp_call)
            mon.start()
            state.whatsapp_monitor = mon
        yield
        state.cancel_pending_shutdown()
        if state.whatsapp_monitor is not None:
            state.whatsapp_monitor.stop()

    app = FastAPI(title="Project Jarvis — On-Demand macOS Assistant", lifespan=lifespan)
    app.state.jarvis = state

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def index_page():
        state.cancel_pending_shutdown()
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/status")
    async def get_status():
        ollama_ok = False
        try:
            async with httpx.AsyncClient(timeout=1.5) as client:
                r = await client.get(f"{state.config.llm.ollama_base_url.rstrip('/')}/api/tags")
                ollama_ok = r.status_code == 200
        except Exception:
            ollama_ok = False

        return {
            "mode": "ACTIVE",
            "follow_up_window_seconds": FOLLOW_UP_WINDOW_SECONDS,
            "stt_engine": f"{state.config.stt.engine} ({state.config.stt.model_size})",
            "tts_engine": state.config.tts.engine,
            "edge_voice": state.config.tts.edge_voice,
            "tts_voice": state.config.tts.macos_voice,
            "llm_provider": state.config.llm.provider,
            "ollama_model": state.config.llm.ollama_model,
            "ollama_online": ollama_ok,
            "auto_shutdown_on_close": state.auto_shutdown_on_close,
            "confirm_destructive_actions": state.config.safety.confirm_destructive_actions,
            "permissions": {
                "accessibility": check_accessibility(),
                "automation": check_automation_system_events(),
                "full_disk_access": check_full_disk_access(),
            },
        }

    @app.post("/api/settings")
    async def update_settings(payload: SettingsUpdatePayload):
        if payload.llm_provider is not None:
            state.config.llm.provider = payload.llm_provider.strip().lower()
        if payload.ollama_model is not None:
            state.config.llm.ollama_model = payload.ollama_model.strip()
        if payload.tts_engine is not None:
            state.config.tts.engine = payload.tts_engine.strip().lower()
        if payload.edge_voice is not None:
            state.config.tts.edge_voice = payload.edge_voice.strip()
        if payload.macos_voice is not None:
            state.config.tts.macos_voice = payload.macos_voice.strip()
        if payload.auto_shutdown_on_close is not None:
            state.auto_shutdown_on_close = bool(payload.auto_shutdown_on_close)
        if payload.confirm_destructive_actions is not None:
            state.config.safety.confirm_destructive_actions = bool(
                payload.confirm_destructive_actions
            )
        return {"ok": True, "status": await get_status()}

    @app.get("/api/contacts")
    async def list_contacts():
        return {"ok": True, "contacts": state.contacts.load()}

    @app.post("/api/contacts")
    async def upsert_contact(payload: ContactPayload):
        updated = state.contacts.upsert(payload.name, payload.phone)
        return {"ok": True, "contacts": updated}

    @app.delete("/api/contacts/{name}")
    async def delete_contact(name: str):
        updated = state.contacts.delete(name)
        return {"ok": True, "contacts": updated}

    @app.post("/api/beacon-close")
    async def beacon_close(payload: Optional[BeaconPayload] = None):
        if payload and payload.navigating:
            return {"ok": True, "action": "navigating_ignored"}
        state.schedule_shutdown_if_idle()
        return {"ok": True, "action": "shutdown_checked"}

    @app.post("/api/shutdown")
    async def manual_shutdown():
        state._terminated = True
        state.cancel_pending_shutdown()
        for ws in list(state.active_sockets):
            try:
                await ws.send_json({"type": "terminated", "reason": "user_terminated"})
                await ws.close()
            except Exception:
                pass
        asyncio.get_running_loop().call_later(0.2, trigger_process_exit)
        return {"ok": True, "message": "Jarvis local bridge terminated."}

    @app.websocket("/ws/jarvis")
    async def ws_jarvis(websocket: WebSocket):
        await websocket.accept()
        state.active_sockets.add(websocket)
        state.cancel_pending_shutdown()

        # 0.45s minimum speech allows quick "Jarvis..." wake calls without missing short words
        segmenter = WebAudioSegmenter(state.config, min_continuous_override=0.45)
        agent = WebAgentEngine(config=state.config)
        listening_active = True
        client_sample_rate = state.config.audio.sample_rate
        follow_up_deadline = 0.0
        processing_lock = asyncio.Lock()

        await websocket.send_json(
            {
                "type": "ready",
                "mode": "ACTIVE",
                "follow_up_window_seconds": FOLLOW_UP_WINDOW_SECONDS,
                "ollama_model": state.config.llm.ollama_model,
            }
        )

        async def _send_step(step_payload: dict[str, Any]) -> None:
            await websocket.send_json({"type": "step", **step_payload})

        async def _run_agent_and_reply(user_text: str) -> None:
            nonlocal follow_up_deadline
            await websocket.send_json({"type": "status", "state": "thinking"})
            turn = await agent.handle_user_input(user_text, on_step=_send_step)
            if turn.sleep_requested:
                follow_up_deadline = 0.0
                synth = await state.voice_auth.synthesize_tts_async(turn.spoken_reply)
                await websocket.send_json(
                    {
                        "type": "standby",
                        "text": turn.spoken_reply,
                        "audio_b64": synth.audio_b64 if synth else None,
                        "mime_type": synth.mime_type if synth else "audio/wav",
                    }
                )
                return

            synth = await state.voice_auth.synthesize_tts_async(turn.spoken_reply)
            # Keep 10-second active follow-up window open after responding
            follow_up_deadline = time.monotonic() + FOLLOW_UP_WINDOW_SECONDS
            print(f'[J.A.R.V.I.S.] "{turn.spoken_reply}"', flush=True)
            await websocket.send_json(
                {
                    "type": "reply",
                    "text": turn.spoken_reply,
                    "audio_b64": synth.audio_b64 if synth else None,
                    "mime_type": synth.mime_type if synth else "audio/wav",
                    "tts_engine": synth.engine if synth else "browser",
                    "route": turn.route,
                    "pause_mic": turn.pause_mic,
                    "follow_up_seconds": FOLLOW_UP_WINDOW_SECONDS,
                    "awaiting_confirmation": turn.awaiting_confirmation,
                    "steps": [
                        {
                            "step_index": s.step_index,
                            "tool_name": s.tool_name,
                            "ok": s.ok,
                            "message": s.message,
                            "latency_ms": round(s.latency_ms, 1),
                        }
                        for s in turn.steps
                    ],
                }
            )
            await websocket.send_json(
                {
                    "type": "status",
                    "state": "listening" if listening_active else "paused",
                }
            )

        async def _handle_voice_utterance(pcm16: np.ndarray) -> None:
            nonlocal follow_up_deadline
            async with processing_lock:
                res = await asyncio.to_thread(
                    state.voice_auth.transcribe_utterance,
                    pcm16,
                    state.config.audio.sample_rate,
                )
                heard = (res.text or "").strip()
                alnum_text = re.sub(r"[^A-Za-z0-9]", "", heard)
                if len(alnum_text) < 3:
                    return

                wake_called, task_after_wake = extract_jarvis_command(heard)

                if wake_called and not task_after_wake:
                    # User said "Jarvis" / "Hey Jarvis" alone -> greet & keep 10s follow-up
                    follow_up_deadline = time.monotonic() + FOLLOW_UP_WINDOW_SECONDS
                    greeting = "At your service, sir."
                    chime_b64 = (
                        base64.b64encode(CHIME_PATH.read_bytes()).decode("ascii")
                        if CHIME_PATH.exists()
                        else None
                    )
                    synth = await state.voice_auth.synthesize_tts_async(greeting)
                    print(f'[Whisper] "{heard}" -> Greeting', flush=True)
                    await websocket.send_json(
                        {
                            "type": "wake_triggered",
                            "heard": heard,
                            "greeting": greeting,
                            "follow_up_seconds": FOLLOW_UP_WINDOW_SECONDS,
                            "chime_b64": chime_b64,
                            "audio_b64": synth.audio_b64 if synth else None,
                            "mime_type": synth.mime_type if synth else "audio/wav",
                        }
                    )
                    return

                command_text = task_after_wake if wake_called else heard
                print(
                    f'[Whisper] "{heard}" -> executing: "{command_text}"',
                    flush=True,
                )
                await websocket.send_json(
                    {"type": "live_transcript", "text": heard}
                )
                await websocket.send_json(
                    {
                        "type": "transcript",
                        "text": command_text,
                        "raw_text": heard,
                        "stt_ms": round(res.stt_ms, 1),
                    }
                )
                await _run_agent_and_reply(command_text)

        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break

                if message.get("bytes") is not None:
                    if not listening_active:
                        continue
                    pcm16 = np.frombuffer(message["bytes"], dtype=np.int16).copy()
                    utt = segmenter.feed(pcm16, sample_rate=client_sample_rate)
                    if utt is not None:
                        await _handle_voice_utterance(utt)
                    continue

                if message.get("text") is not None:
                    import json

                    data = json.loads(message["text"])
                    mtype = data.get("type", "")

                    if mtype == "ping":
                        await websocket.send_json({"type": "pong"})
                    elif mtype == "set_sample_rate":
                        sr = int(data.get("sample_rate", 16000))
                        if 8000 <= sr <= 96000:
                            client_sample_rate = sr
                    elif mtype == "set_listening":
                        listening_active = bool(data.get("active", True))
                        segmenter.reset()
                        await websocket.send_json(
                            {
                                "type": "status",
                                "state": "listening" if listening_active else "paused",
                            }
                        )
                    elif mtype == "flush_audio":
                        utt = segmenter.flush()
                        if utt is not None:
                            await _handle_voice_utterance(utt)
                    elif mtype == "text_command":
                        txt = str(data.get("text", "")).strip()
                        if txt:
                            async with processing_lock:
                                await _run_agent_and_reply(txt)
                    elif mtype == "clear_history":
                        agent.clear_history()
                        follow_up_deadline = 0.0
                        await websocket.send_json({"type": "history_cleared"})
                    elif mtype == "terminate":
                        state._terminated = True
                        segmenter.reset()
                        await websocket.send_json(
                            {"type": "terminated", "reason": "user_terminated"}
                        )
                        asyncio.get_running_loop().call_later(0.15, trigger_process_exit)
                        break

        except WebSocketDisconnect:
            pass
        finally:
            segmenter.reset()
            state.active_sockets.discard(websocket)
            state.schedule_shutdown_if_idle()

    return app


def _schedule_browser_open(url: str, delay_sec: float = 1.5) -> None:
    def _open():
        time.sleep(delay_sec)
        subprocess.run(["open", url], check=False)

    threading.Thread(target=_open, daemon=True, name="jarvis-open-browser").start()


def main() -> None:
    parser = argparse.ArgumentParser(description="Project Jarvis On-Demand Web Server")
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Automatically open http://localhost:8000 1.5s after binding.",
    )
    parser.add_argument("--host", default=os.environ.get("JARVIS_WEB_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("JARVIS_WEB_PORT", "8000")))
    args = parser.parse_args()

    kill_stale_port(args.port)

    application = create_app(warmup_models=True)
    if args.open_browser:
        _schedule_browser_open(f"http://localhost:{args.port}", delay_sec=1.5)

    uvicorn.run(application, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
else:
    app = create_app(warmup_models=True)
