"""Wake-word detection and lightweight audio ring buffer for Project Jarvis.

Continuously processes 80 ms (1280-sample @ 16 kHz) audio frames in a bounded
ring buffer with minimal CPU overhead until the wake phrase ("Jarvis") fires.
"""

from __future__ import annotations

import collections
import logging
import os
import time
from dataclasses import dataclass
from typing import Deque, Optional, Protocol

import numpy as np

from config.settings import AudioConfig, JarvisConfig, WakeWordConfig, load_config
from core.biometrics import compute_rms_dbfs, pcm16_to_float32

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WakeDetection:
    detected: bool
    score: float
    keyword: str


class WakeWordBackend(Protocol):
    """Protocol for wake-word detection engines."""

    name: str

    def process_frame(self, pcm16: np.ndarray) -> WakeDetection: ...

    def reset(self) -> None: ...


class OpenWakeWordBackend:
    """openWakeWord backend using built-in 'hey_jarvis' (or custom) ONNX/TFLite model."""

    name = "openwakeword"

    def __init__(self, model_name: str = "hey_jarvis", threshold: float = 0.50) -> None:
        import openwakeword
        from openwakeword.model import Model

        # Ensure default models are downloaded if not yet cached
        try:
            openwakeword.utils.download_models([model_name])
        except Exception as exc:
            logger.debug("openWakeWord model download check skipped/warned: %s", exc)

        self.model_name = model_name
        self.threshold = threshold
        try:
            self._model = Model(wakeword_models=[model_name], inference_framework="onnx")
        except Exception:
            # Fallback to default inference framework if onnxruntime variant differs
            self._model = Model(wakeword_models=[model_name])
        logger.info("Initialized openWakeWord model '%s' (threshold=%.2f).", model_name, threshold)

    def process_frame(self, pcm16: np.ndarray) -> WakeDetection:
        prediction = self._model.predict(pcm16)
        best_label = self.model_name
        best_score = 0.0
        for key, val in prediction.items():
            score = float(val)
            if self.model_name in key or score > best_score:
                if score > best_score:
                    best_score = score
                    best_label = key
        return WakeDetection(
            detected=best_score >= self.threshold,
            score=best_score,
            keyword=best_label,
        )

    def reset(self) -> None:
        if hasattr(self._model, "reset"):
            self._model.reset()


class PorcupineBackend:
    """Picovoice Porcupine wake-word backend (optional when PICOVOICE_ACCESS_KEY is set)."""

    name = "pvporcupine"

    def __init__(self, access_key: str, keyword: str = "jarvis") -> None:
        import pvporcupine

        self._porcupine = pvporcupine.create(access_key=access_key, keywords=[keyword])
        self.keyword = keyword
        self.frame_length = int(self._porcupine.frame_length)
        self._pending = np.zeros(0, dtype=np.int16)
        logger.info("Initialized Porcupine wake word '%s'.", keyword)

    def process_frame(self, pcm16: np.ndarray) -> WakeDetection:
        self._pending = np.concatenate([self._pending, pcm16])
        detected = False
        while self._pending.size >= self.frame_length:
            chunk = self._pending[: self.frame_length]
            self._pending = self._pending[self.frame_length :]
            idx = self._porcupine.process(chunk)
            if idx >= 0:
                detected = True
        return WakeDetection(
            detected=detected,
            score=1.0 if detected else 0.0,
            keyword=self.keyword,
        )

    def reset(self) -> None:
        self._pending = np.zeros(0, dtype=np.int16)

    def close(self) -> None:
        if self._porcupine is not None:
            self._porcupine.delete()
            self._porcupine = None


def create_wake_word_backend(cfg: WakeWordConfig) -> WakeWordBackend:
    """Create configured wake-word backend (openWakeWord default, Porcupine optional)."""
    engine = cfg.engine.lower()
    if engine == "pvporcupine":
        key = os.environ.get(cfg.porcupine_access_key_env, "").strip()
        if key:
            return PorcupineBackend(access_key=key, keyword=cfg.porcupine_keyword)
        logger.warning(
            "Porcupine selected but %s is unset; falling back to openWakeWord.",
            cfg.porcupine_access_key_env,
        )
    return OpenWakeWordBackend(model_name=cfg.model_name, threshold=cfg.threshold)


class AudioRingBuffer:
    """Fixed-capacity ring buffer storing recent 16 kHz int16 audio frames."""

    def __init__(self, max_samples: int) -> None:
        self.max_samples = max(1, int(max_samples))
        self._chunks: Deque[np.ndarray] = collections.deque()
        self._total_samples = 0

    def append(self, pcm16: np.ndarray) -> None:
        arr = np.asarray(pcm16, dtype=np.int16).reshape(-1)
        if arr.size == 0:
            return
        self._chunks.append(arr)
        self._total_samples += arr.size
        while self._total_samples > self.max_samples and self._chunks:
            oldest = self._chunks.popleft()
            self._total_samples -= oldest.size

    def get_last_samples(self, num_samples: int) -> np.ndarray:
        if self._total_samples == 0:
            return np.zeros(0, dtype=np.int16)
        combined = np.concatenate(list(self._chunks))
        if combined.size <= num_samples:
            return combined
        return combined[-num_samples:]

    def clear(self) -> None:
        self._chunks.clear()
        self._total_samples = 0


class SpeechActivityDetector:
    """Low-overhead VAD using webrtcvad with RMS energy fallback."""

    def __init__(self, audio_cfg: AudioConfig) -> None:
        self.sample_rate = audio_cfg.sample_rate
        self.energy_threshold_dbfs = audio_cfg.energy_threshold_dbfs
        self._vad = None
        try:
            import webrtcvad

            self._vad = webrtcvad.Vad(audio_cfg.vad_aggressiveness)
        except Exception as exc:
            logger.warning("webrtcvad unavailable (%s); using RMS energy VAD.", exc)

    def is_speech(self, pcm16: np.ndarray) -> bool:
        """Return True if the audio frame contains active speech."""
        if pcm16.size == 0:
            return False

        # Quick RMS gate prevents false positives from digital silence
        rms_db = compute_rms_dbfs(pcm16_to_float32(pcm16))
        if rms_db < self.energy_threshold_dbfs:
            return False

        if self._vad is None:
            return True

        # WebRTC VAD expects 10ms, 20ms, or 30ms frames (320 samples = 20ms @ 16kHz)
        subframe_len = int(self.sample_rate * 0.02)
        speech_hits = 0
        total_subframes = 0
        for start in range(0, pcm16.size - subframe_len + 1, subframe_len):
            sub = pcm16[start : start + subframe_len]
            total_subframes += 1
            if self._vad.is_speech(sub.tobytes(), self.sample_rate):
                speech_hits += 1

        if total_subframes == 0:
            return rms_db >= self.energy_threshold_dbfs
        return speech_hits >= max(1, total_subframes // 3)


class WakeWordDetector:
    """Stateful wake-word detector with patience and cooldown gating."""

    def __init__(self, config: Optional[JarvisConfig] = None, backend: Optional[WakeWordBackend] = None) -> None:
        self.config = config or load_config()
        self.backend = backend or create_wake_word_backend(self.config.wake_word)
        self._consecutive_hits = 0
        self._last_trigger_ts = 0.0

    def feed(self, pcm16: np.ndarray) -> WakeDetection:
        res = self.backend.process_frame(pcm16)
        now = time.monotonic()
        if res.detected:
            self._consecutive_hits += 1
        else:
            self._consecutive_hits = 0

        if (
            self._consecutive_hits >= self.config.wake_word.patience_frames
            and (now - self._last_trigger_ts) >= self.config.wake_word.cooldown_seconds
        ):
            self._consecutive_hits = 0
            self._last_trigger_ts = now
            self.backend.reset()
            return WakeDetection(detected=True, score=res.score, keyword=res.keyword)

        return WakeDetection(detected=False, score=res.score, keyword=res.keyword)

    def reset(self) -> None:
        self._consecutive_hits = 0
        self.backend.reset()
