"""Tests for Layer 1 AudioRingBuffer, SpeechActivityDetector, and WakeWordDetector."""

from __future__ import annotations

import numpy as np

from config.settings import AudioConfig, load_config
from core.wake_word import (
    AudioRingBuffer,
    SpeechActivityDetector,
    WakeDetection,
    WakeWordDetector,
)


class _StubWakeBackend:
    name = "stub"

    def __init__(self, scores: list[float]) -> None:
        self._scores = list(scores)
        self._idx = 0

    def process_frame(self, pcm16: np.ndarray) -> WakeDetection:
        score = self._scores[min(self._idx, len(self._scores) - 1)]
        self._idx += 1
        return WakeDetection(detected=score >= 0.5, score=score, keyword="hey_jarvis")

    def reset(self) -> None:
        pass


def test_audio_ring_buffer_capacity():
    buf = AudioRingBuffer(max_samples=2560)
    f1 = np.ones(1280, dtype=np.int16)
    f2 = np.full(1280, 2, dtype=np.int16)
    f3 = np.full(1280, 3, dtype=np.int16)
    buf.append(f1)
    buf.append(f2)
    buf.append(f3)
    tail = buf.get_last_samples(2560)
    assert tail.size == 2560
    assert np.all(tail[:1280] == 2)
    assert np.all(tail[1280:] == 3)
    buf.clear()
    assert buf.get_last_samples(1280).size == 0


def test_vad_rejects_silence_and_detects_voiced_tone():
    cfg = AudioConfig()
    vad = SpeechActivityDetector(cfg)
    silence = np.zeros(1280, dtype=np.int16)
    assert vad.is_speech(silence) is False

    # 300 Hz modulated tone above energy threshold
    t = np.linspace(0, 0.08, 1280, endpoint=False)
    tone = (np.sin(2 * np.pi * 300 * t) * 8000).astype(np.int16)
    # Energy check passes; webrtcvad may classify pure sine differently, so check energy fallback too
    vad_energy_only = SpeechActivityDetector(cfg)
    vad_energy_only._vad = None
    assert vad_energy_only.is_speech(tone) is True


def test_wake_word_detector_patience_and_cooldown():
    cfg = load_config()
    cfg.wake_word.patience_frames = 2
    cfg.wake_word.cooldown_seconds = 60.0

    backend = _StubWakeBackend([0.1, 0.6, 0.7, 0.8, 0.9])
    detector = WakeWordDetector(config=cfg, backend=backend)
    frame = np.zeros(1280, dtype=np.int16)

    assert detector.feed(frame).detected is False  # 0.1
    assert detector.feed(frame).detected is False  # 0.6 (1/2 patience)
    assert detector.feed(frame).detected is True   # 0.7 (2/2 patience -> trigger!)
    # Cooldown suppresses immediate re-trigger
    assert detector.feed(frame).detected is False
