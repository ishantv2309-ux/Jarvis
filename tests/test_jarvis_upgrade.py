"""Tests for call-by-name wake gate, 10s follow-up window, and British J.A.R.V.I.S. TTS."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path

from fastapi.testclient import TestClient

from agent_tools import WebAgentEngine
from config.settings import PROJECT_ROOT, load_config
from core.standby_listener import (
    extract_jarvis_command,
    matches_sleep_phrase,
    matches_wake_phrase,
)
from server import create_app
from tts_jarvis import JarvisTTS, ensure_boot_chime_wav


def test_wake_extraction_and_sleep_phrases():
    assert matches_wake_phrase("Jarvis wake up") is True
    assert matches_wake_phrase("Hey Jarvis, open Safari") is True
    assert matches_wake_phrase("What time is the meeting today?") is False

    assert extract_jarvis_command("Jarvis, open Visual Studio Code") == (
        True,
        "open Visual Studio Code",
    )
    assert extract_jarvis_command("Hey Jarvis play lo-fi beats on YouTube") == (
        True,
        "play lo-fi beats on YouTube",
    )
    assert extract_jarvis_command("Wake up Jarvis") == (True, "")
    assert extract_jarvis_command("Just random room chatter") == (False, "")

    assert matches_sleep_phrase("Go to sleep") is True
    assert matches_sleep_phrase("Jarvis go to sleep") is True
    assert matches_sleep_phrase("Standby") is True
    assert matches_sleep_phrase("Open Terminal") is False


def test_boot_chime_and_jarvis_tts_synthesis(tmp_path: Path):
    chime_path = ensure_boot_chime_wav(tmp_path / "chime.wav")
    assert chime_path.exists()
    assert chime_path.read_bytes()[:4] == b"RIFF"

    cfg = load_config()
    tts = JarvisTTS(config=cfg)
    synth = tts.synthesize_sync("At your service, sir. Systems online.")
    assert synth is not None
    assert len(synth.raw_bytes) > 1000
    assert synth.mime_type in ("audio/mpeg", "audio/wav")

    cfg.tts.engine = "say"
    tts_offline = JarvisTTS(config=cfg)
    synth_off = tts_offline.synthesize_sync("Entering standby mode, sir.")
    assert synth_off is not None
    assert synth_off.mime_type == "audio/wav"
    assert synth_off.raw_bytes[:4] == b"RIFF"


def test_agent_sleep_command_and_standby_websocket(tmp_path: Path):
    cfg = load_config()
    cfg.safety.audit_log_path = str(tmp_path / "audit.log")
    engine = WebAgentEngine(config=cfg)

    async def _run():
        res = await engine.handle_user_input("Go to sleep")
        assert res.sleep_requested is True
        assert "standby" in res.spoken_reply.lower()
        assert "sir" in res.spoken_reply.lower()

    asyncio.run(_run())

    app = create_app(config=cfg, warmup_models=False, enable_whatsapp_monitor=False)
    with TestClient(app) as client:
        st = client.get("/api/status").json()
        assert st["mode"] == "ACTIVE"
        assert st["follow_up_window_seconds"] == 10.0

        with client.websocket_connect("/ws/jarvis") as ws:
            ready = ws.receive_json()
            assert ready["type"] == "ready"

            ws.send_json({"type": "text_command", "text": "Go to sleep"})
            sb = None
            for _ in range(5):
                ev = ws.receive_json()
                if ev["type"] == "standby":
                    sb = ev
                    break
            assert sb is not None
            assert sb["audio_b64"] is not None
            assert len(base64.b64decode(sb["audio_b64"])) > 500


def test_standalone_launchers_and_desktop_app_exist_and_executable():
    import os

    files = [
        PROJECT_ROOT / "Launch_Jarvis.command",
        PROJECT_ROOT / "install_autostart.sh",
        PROJECT_ROOT / "uninstall_autostart.sh",
        Path("/Users/ishantverma/Desktop/Jarvis.app/Contents/MacOS/Jarvis"),
    ]
    for f in files:
        assert f.exists(), f"Missing {f}"
        assert os.access(f, os.X_OK), f"Not executable: {f}"

    assert (PROJECT_ROOT / "launchd" / "com.jarvis.assistant.plist").exists()
    assert (Path("/Users/ishantverma/Desktop/Jarvis.app/Contents/Info.plist")).exists()
