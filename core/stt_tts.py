"""Speech-to-Text (faster-whisper) and Text-to-Speech (macOS say / Kokoro) for Project Jarvis."""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from config.settings import JarvisConfig, load_config
from core.biometrics import pcm16_to_float32

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str
    duration_sec: float
    latency_ms: float


class SpeechTranscriber:
    """Local faster-whisper speech-to-text transcriber (runs ONLY on verified audio)."""

    def __init__(self, config: Optional[JarvisConfig] = None) -> None:
        self.config = config or load_config()
        self._model = None
        self._lock = threading.Lock()

    def _ensure_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from faster_whisper import WhisperModel

                    stt_cfg = self.config.stt
                    logger.info(
                        "Loading faster-whisper '%s' (%s/%s)...",
                        stt_cfg.model_size,
                        stt_cfg.device,
                        stt_cfg.compute_type,
                    )
                    self._model = WhisperModel(
                        stt_cfg.model_size,
                        device=stt_cfg.device,
                        compute_type=stt_cfg.compute_type,
                    )
        return self._model

    def warmup(self) -> None:
        """Pre-load model weights so first voice command has minimal latency."""
        self._ensure_model()

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> TranscriptionResult:
        t0 = time.perf_counter()
        f32 = pcm16_to_float32(audio)
        if f32.size == 0:
            return TranscriptionResult(text="", language=self.config.stt.language, duration_sec=0.0, latency_ms=0.0)

        model = self._ensure_model()
        segments, info = model.transcribe(
            f32,
            language=self.config.stt.language,
            beam_size=self.config.stt.beam_size,
            vad_filter=True,
            condition_on_previous_text=False,
            initial_prompt=self.config.stt.initial_prompt or None,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return TranscriptionResult(
            text=text,
            language=getattr(info, "language", self.config.stt.language),
            duration_sec=f32.size / float(sample_rate),
            latency_ms=elapsed_ms,
        )


class SpeechSynthesizer:
    """Low-latency spoken feedback via macOS `say` or optional Kokoro TTS."""

    def __init__(
        self,
        config: Optional[JarvisConfig] = None,
        on_speaking_change: Optional[Callable[[bool], None]] = None,
    ) -> None:
        self.config = config or load_config()
        self._on_speaking_change = on_speaking_change
        self._speak_lock = threading.Lock()
        self._kokoro_pipeline = None

    def set_speaking_callback(self, cb: Optional[Callable[[bool], None]]) -> None:
        self._on_speaking_change = cb

    def _notify(self, speaking: bool) -> None:
        if self._on_speaking_change is not None:
            try:
                self._on_speaking_change(speaking)
            except Exception:
                pass

    def speak(self, text: str, block: bool = True) -> None:
        """Speak text aloud while notifying audio gate to suppress self-triggering."""
        cleaned = (text or "").strip()
        if not cleaned:
            return

        def _run() -> None:
            with self._speak_lock:
                self._notify(True)
                try:
                    if self.config.tts.engine.lower() == "kokoro":
                        if self._speak_kokoro(cleaned):
                            return
                    self._speak_macos_say(cleaned)
                finally:
                    # Brief tail delay so room echo doesn't re-trigger VAD
                    time.sleep(0.18)
                    self._notify(False)

        if block:
            _run()
        else:
            threading.Thread(target=_run, daemon=True, name="jarvis-tts").start()

    def _speak_macos_say(self, text: str) -> None:
        cmd = [
            "say",
            "-v",
            self.config.tts.macos_voice,
            "-r",
            str(self.config.tts.macos_rate),
            text,
        ]
        try:
            subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:
            logger.error("macOS `say` failed: %s", exc)

    def _speak_kokoro(self, text: str) -> bool:
        try:
            if self._kokoro_pipeline is None:
                from kokoro import KPipeline  # type: ignore

                self._kokoro_pipeline = KPipeline(lang_code=self.config.tts.kokoro_lang)
            import pyaudio

            pa = pyaudio.PyAudio()
            stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=24000, output=True)
            try:
                for _, _, audio in self._kokoro_pipeline(text, voice=self.config.tts.kokoro_voice):
                    stream.write(np.asarray(audio, dtype=np.float32).tobytes())
            finally:
                stream.stop_stream()
                stream.close()
                pa.terminate()
            return True
        except Exception as exc:
            logger.warning("Kokoro TTS unavailable (%s); falling back to macOS `say`.", exc)
            return False
