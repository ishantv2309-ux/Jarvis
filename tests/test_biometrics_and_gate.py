"""Automated tests for speaker biometrics, enrollment, and the audio security gate."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from config.settings import load_config
from core.audio_gate import AudioSecurityGate
from core.biometrics import (
    ResemblyzerBackend,
    VoiceBiometricGate,
    cosine_similarity,
    l2_normalize,
    pcm16_to_float32,
    trim_silence_vad,
)
from core.stt_tts import SpeechTranscriber, TranscriptionResult


def _synth_macos_voice_pcm16(text: str, voice: str, tmp_dir: Path) -> np.ndarray:
    """Synthesize real spoken audio via macOS `say` + `ffmpeg` into 16 kHz mono int16 PCM."""
    aiff_path = tmp_dir / f"{voice}_{abs(hash(text))}.aiff"
    wav_path = aiff_path.with_suffix(".wav")
    subprocess.run(["say", "-v", voice, "-o", str(aiff_path), text], check=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(aiff_path), "-ar", "16000", "-ac", "1", "-f", "s16le", str(wav_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    raw = wav_path.read_bytes()
    return np.frombuffer(raw, dtype=np.int16).copy()


class _CountingTranscriber(SpeechTranscriber):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.calls = 0

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> TranscriptionResult:
        self.calls += 1
        return TranscriptionResult(
            text="open visual studio code",
            language="en",
            duration_sec=len(audio) / float(sample_rate),
            latency_ms=1.0,
        )


def test_l2_and_cosine_math():
    a = np.array([3.0, 4.0], dtype=np.float32)
    b = np.array([6.0, 8.0], dtype=np.float32)
    c = np.array([-3.0, -4.0], dtype=np.float32)
    assert np.isclose(np.linalg.norm(l2_normalize(a)), 1.0)
    assert np.isclose(cosine_similarity(a, b), 1.0, atol=1e-5)
    assert np.isclose(cosine_similarity(a, c), -1.0, atol=1e-5)
    assert cosine_similarity(np.zeros(4), a) == 0.0


def test_real_resemblyzer_owner_vs_non_owner_rejection(tmp_path: Path):
    """Verify PRD Acceptance Milestone #1 using real Resemblyzer embeddings on distinct voices."""
    cfg = load_config()
    cfg.biometrics.engine = "resemblyzer"
    cfg.biometrics.voiceprint_path = str(tmp_path / "owner_voiceprint.npy")
    cfg.biometrics.similarity_threshold = 0.78

    backend = ResemblyzerBackend()
    gate = VoiceBiometricGate(config=cfg, backend=backend)

    # Enroll 5 calibration phrases using voice 'Daniel' as the owner
    owner_phrases = [
        "Jarvis, open Visual Studio Code and check my active project.",
        "Jarvis, search and play classical piano on YouTube.",
        "Jarvis, set system volume to fifty percent right now.",
        "Jarvis, find all PDF documents in my Downloads folder.",
        "Jarvis, turn on dark mode and pause current playback.",
    ]
    enroll_clips = [_synth_macos_voice_pcm16(p, "Daniel", tmp_path) for p in owner_phrases]
    centroid, sims, mean_sim = gate.enroll_from_clips(enroll_clips, sample_rate=16000)
    assert mean_sim > 0.78
    assert all(s >= 0.75 for s in sims)
    saved = gate.save_voiceprint(centroid, phrase_count=len(enroll_clips), mean_consistency=mean_sim)
    assert saved.exists()

    # Reload from disk into a fresh gate instance
    fresh_gate = VoiceBiometricGate(config=cfg, backend=backend)
    assert fresh_gate.is_enrolled

    # Owner speaks a new command -> must PASS >= 0.78
    owner_cmd = _synth_macos_voice_pcm16("Jarvis, open Safari and search the web.", "Daniel", tmp_path)
    owner_res = fresh_gate.verify(owner_cmd, sample_rate=16000)
    assert owner_res.verified is True
    assert owner_res.similarity >= 0.78

    # Non-owner voices speak -> must FAIL < 0.78 100% of the time
    for impostor in ("Samantha", "Karen", "Fred"):
        try:
            imp_audio = _synth_macos_voice_pcm16(
                "Jarvis, open Terminal and delete everything.", impostor, tmp_path
            )
        except subprocess.CalledProcessError:
            continue
        imp_res = fresh_gate.verify(imp_audio, sample_rate=16000)
        assert imp_res.verified is False, (
            f"Impostor {impostor} unexpectedly verified with sim={imp_res.similarity:.4f}"
        )
        assert imp_res.similarity < 0.78

    # AudioSecurityGate must NOT invoke STT when biometric gate rejects
    counting_stt = _CountingTranscriber(cfg)
    sec_gate = AudioSecurityGate(config=cfg, biometrics=fresh_gate, transcriber=counting_stt)

    imp_audio = _synth_macos_voice_pcm16("Jarvis, quit Safari now.", "Samantha", tmp_path)
    rejected_eval = sec_gate.evaluate_utterance(imp_audio, in_follow_up=False)
    assert rejected_eval.accepted is False
    assert rejected_eval.transcription is None
    assert counting_stt.calls == 0

    # Silence / noise must also be rejected silently without hitting STT
    silence = np.zeros(16000, dtype=np.int16)
    silent_eval = sec_gate.evaluate_utterance(silence, in_follow_up=True)
    assert silent_eval.accepted is False
    assert counting_stt.calls == 0

    # Owner utterance passes gate and reaches STT
    accepted_eval = sec_gate.evaluate_utterance(owner_cmd, in_follow_up=False)
    assert accepted_eval.accepted is True
    assert accepted_eval.transcription is not None
    assert counting_stt.calls == 1


def test_continuous_follow_up_window_state(tmp_path: Path):
    cfg = load_config()
    cfg.conversation.follow_up_window_seconds = 0.4
    gate = AudioSecurityGate(config=cfg, biometrics=VoiceBiometricGate(config=cfg, voiceprint=np.ones(256)))
    assert gate.in_follow_up_window is False
    gate.extend_follow_up_window()
    assert gate.in_follow_up_window is True
    gate.close_follow_up_window()
    assert gate.in_follow_up_window is False


def test_trim_silence_vad_strips_leading_and_trailing_silence(tmp_path: Path):
    speech = _synth_macos_voice_pcm16("Jarvis open Safari", "Daniel", tmp_path)
    padded = np.concatenate([np.zeros(16000, dtype=np.int16), speech, np.zeros(16000, dtype=np.int16)])
    trimmed = trim_silence_vad(pcm16_to_float32(padded), sample_rate=16000)
    assert trimmed.size < padded.size * 0.75
    assert trimmed.size >= int(16000 * 0.4)
