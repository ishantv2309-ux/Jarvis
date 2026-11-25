"""Automated tests for the On-Demand Web Jarvis FastAPI bridge, WebSockets, and multi-step agent."""

from __future__ import annotations

import asyncio
import base64
import subprocess
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from agent_tools import WebAgentEngine, fast_path_clause, split_multi_step_clauses
from config.settings import load_config
from core.biometrics import (
    SPEAKER_THRESHOLD,
    ResemblyzerBackend,
    compute_rms_dbfs,
    normalize_audio_f32,
    resample_to_16k_f32,
)
from server import create_app
from voice_auth import WebAudioSegmenter


def _synth_pcm16(text: str, voice: str, tmp_dir: Path) -> np.ndarray:
    aiff = tmp_dir / f"{voice}_{abs(hash(text))}.aiff"
    wav = aiff.with_suffix(".wav")
    subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(aiff), "-ar", "16000", "-ac", "1", "-f", "s16le", str(wav)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return np.frombuffer(wav.read_bytes(), dtype=np.int16).copy()


def test_multi_step_clause_splitting_and_folder_routing():
    clauses = split_multi_step_clauses(
        "Jarvis, open VS Code, create a new project folder named AlphaApp on Desktop, and play lo-fi beats on YouTube"
    )
    assert len(clauses) == 3
    assert fast_path_clause(clauses[0])[0][0] == "manage_application"
    mkdir_call = fast_path_clause(clauses[1])
    assert mkdir_call[0][0] == "manage_files"
    assert mkdir_call[0][1]["action"] == "mkdir"
    assert mkdir_call[0][1]["source"].endswith("Desktop/AlphaApp")
    assert fast_path_clause(clauses[2])[0][0] == "youtube_search_and_play"


def test_web_destructive_confirmation_yes_confirm(tmp_path: Path):
    target = tmp_path / "remove_me.txt"
    target.write_text("temp", encoding="utf-8")

    cfg = load_config()
    cfg.safety.audit_log_path = str(tmp_path / "audit.log")
    engine = WebAgentEngine(config=cfg)

    async def _run():
        t1 = await engine.handle_user_input(f"delete {target}")
        assert t1.awaiting_confirmation is True
        assert target.exists()

        t2 = await engine.handle_user_input("Yes, confirm")
        assert t2.awaiting_confirmation is False
        assert not target.exists()

    asyncio.run(_run())


def test_vad_ignores_short_noise_under_1_2s_and_keeps_hangover(tmp_path: Path):
    cfg = load_config()
    assert SPEAKER_THRESHOLD == 0.60
    assert cfg.biometrics.similarity_threshold == 0.60

    seg = WebAudioSegmenter(cfg)
    # 0.5s short burst -> must be silently ignored (< 1.2s continuous speech)
    short_speech = _synth_pcm16("Hey.", "Daniel", tmp_path)[: int(16000 * 0.5)]
    chunk = 640
    emitted = []
    for i in range(0, len(short_speech), chunk):
        out = seg.feed(short_speech[i : i + chunk])
        if out is not None:
            emitted.append(out)
    for _ in range(30):
        out = seg.feed(np.zeros(chunk, dtype=np.int16))
        if out is not None:
            emitted.append(out)
    assert emitted == []

    # >= 1.5s continuous speech -> must emit cleanly with 400ms trailing hangover
    long_speech = _synth_pcm16(
        "Jarvis open Visual Studio Code and play relaxing music on YouTube right now.",
        "Daniel",
        tmp_path,
    )
    for i in range(0, len(long_speech), chunk):
        out = seg.feed(long_speech[i : i + chunk])
        if out is not None:
            emitted.append(out)
    for _ in range(35):
        out = seg.feed(np.zeros(chunk, dtype=np.int16))
        if out is not None:
            emitted.append(out)
    assert len(emitted) == 1
    assert emitted[0].size >= int(16000 * 1.4)


def test_audio_resampling_and_normalization():
    # 48 kHz quiet sine -> 16 kHz normalized float32
    t = np.linspace(0, 1.0, 48000, endpoint=False, dtype=np.float32)
    quiet_48k = 0.01 * np.sin(2 * np.pi * 220 * t)
    res_16k = resample_to_16k_f32(quiet_48k, sample_rate=48000)
    assert res_16k.size == 16000
    normed = normalize_audio_f32(res_16k, target_rms_dbfs=-20.0)
    assert -22.0 <= compute_rms_dbfs(normed) <= -18.0


def test_fastapi_routes_open_voice_websocket_and_tts(tmp_path: Path):
    cfg = load_config()
    cfg.safety.audit_log_path = str(tmp_path / "audit.log")

    app = create_app(
        config=cfg,
        warmup_models=False,
        enable_standby_listener=False,
        enable_whatsapp_monitor=False,
    )

    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        status = client.get("/api/status").json()
        assert status["mode"] == "ACTIVE"
        assert "voiceprint_enrolled" not in status

        # Test settings update without voice-lock
        upd = client.post("/api/settings", json={"tts_engine": "say"})
        assert upd.status_code == 200
        assert upd.json()["status"]["tts_engine"] == "say"

        # Test Direct Listening mode (executes with or without "Jarvis" prefix, strips wake prefix)
        with client.websocket_connect("/ws/jarvis") as ws:
            ready = ws.receive_json()
            assert ready["type"] == "ready"
            assert ready["mode"] == "ACTIVE"

            # 1. Direct sentence WITHOUT "Jarvis" -> executes immediately (no ignored gate)
            direct_pcm = _synth_pcm16(
                "What window is active on my Mac right now?",
                "Samantha",
                tmp_path,
            )
            for i in range(0, len(direct_pcm), 640):
                ws.send_bytes(direct_pcm[i : i + 640].tobytes())
            for _ in range(35):
                ws.send_bytes(np.zeros(640, dtype=np.int16).tobytes())

            direct_reply = None
            for _ in range(12):
                ev = ws.receive_json()
                assert ev["type"] != "ignored"
                if ev["type"] == "reply":
                    direct_reply = ev
                    break
            assert direct_reply is not None
            assert len(direct_reply["text"]) > 0
            audio_bytes = base64.b64decode(direct_reply["audio_b64"])
            assert audio_bytes[:4] == b"RIFF"

            # 2. Sentence WITH "Hey Jarvis" prefix -> strips prefix cleanly and executes
            wake_pcm = _synth_pcm16(
                "Hey Jarvis, what apps are currently running on my Mac?",
                "Samantha",
                tmp_path,
            )
            for i in range(0, len(wake_pcm), 640):
                ws.send_bytes(wake_pcm[i : i + 640].tobytes())
            for _ in range(35):
                ws.send_bytes(np.zeros(640, dtype=np.int16).tobytes())

            transcript_txt = ""
            wake_reply = None
            for _ in range(12):
                ev = ws.receive_json()
                if ev["type"] == "transcript":
                    transcript_txt = ev["text"]
                if ev["type"] == "reply":
                    wake_reply = ev
                    break
            assert wake_reply is not None
            assert not transcript_txt.lower().startswith("hey jarvis")

        b_nav = client.post("/api/beacon-close", json={"reason": "tab_closed", "navigating": True})
        assert b_nav.json()["action"] == "navigating_ignored"
