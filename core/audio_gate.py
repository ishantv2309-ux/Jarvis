"""Core Audio & Biometric Gate (Layer 1 + Layer 2 + Layer 3 STT) for Project Jarvis.

Enforces the Core Security Mandate:
- Continuously monitors microphone via lightweight ring buffer + openWakeWord.
- On wake word detection ("Jarvis") OR active 10-second continuous follow-up window:
  1. Captures the spoken command via VAD speech end-pointing.
  2. Extracts speaker embedding and verifies cosine similarity >= 0.78 against
     owner_voiceprint.npy.
  3. Unrecognized voices or background audio are discarded SILENTLY before STT
     or system execution ever sees them.
  4. Verified commands are transcribed with faster-whisper and dispatched.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from config.settings import JarvisConfig, load_config
from core.biometrics import BiometricResult, VoiceBiometricGate
from core.stt_tts import SpeechTranscriber, TranscriptionResult
from core.wake_word import (
    AudioRingBuffer,
    SpeechActivityDetector,
    WakeWordDetector,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GateEvaluation:
    """Structured result from evaluating one captured utterance through the security gate."""

    accepted: bool
    biometric: BiometricResult
    transcription: Optional[TranscriptionResult]
    in_follow_up_window: bool


class AudioSecurityGate:
    """Real-time audio capture, wake-word, biometric lock, and STT pipeline."""

    def __init__(
        self,
        config: Optional[JarvisConfig] = None,
        biometrics: Optional[VoiceBiometricGate] = None,
        wake_detector: Optional[WakeWordDetector] = None,
        transcriber: Optional[SpeechTranscriber] = None,
        on_command: Optional[Callable[[GateEvaluation], None]] = None,
        on_state_change: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.config = config or load_config()
        self.biometrics = biometrics or VoiceBiometricGate(config=self.config)
        self.wake_detector = wake_detector
        self.transcriber = transcriber or SpeechTranscriber(config=self.config)
        self.vad = SpeechActivityDetector(self.config.audio)
        self.ring_buffer = AudioRingBuffer(
            max_samples=int(self.config.audio.sample_rate * self.config.audio.ring_buffer_seconds)
        )

        self._on_command = on_command
        self._on_state_change = on_state_change

        self._stop_event = threading.Event()
        self._muted = threading.Event()
        self._tts_speaking = threading.Event()
        self._follow_up_deadline = 0.0
        self._stream = None
        self._pa = None

    def set_tts_speaking(self, speaking: bool) -> None:
        """Pause mic ingestion while Jarvis speaks so TTS never self-triggers."""
        if speaking:
            self._tts_speaking.set()
        else:
            self.ring_buffer.clear()
            if self.wake_detector is not None:
                self.wake_detector.reset()
            self._tts_speaking.clear()

    def set_muted(self, muted: bool) -> None:
        if muted:
            self._muted.set()
            self._emit_state("muted")
        else:
            self._muted.clear()
            self._emit_state("listening")

    @property
    def is_muted(self) -> bool:
        return self._muted.is_set()

    @property
    def in_follow_up_window(self) -> bool:
        return time.monotonic() < self._follow_up_deadline

    def extend_follow_up_window(self, seconds: Optional[float] = None) -> None:
        """Open/refresh the 10-second continuous conversation window (FR-1)."""
        dur = seconds if seconds is not None else self.config.conversation.follow_up_window_seconds
        self._follow_up_deadline = time.monotonic() + float(dur)
        self._emit_state("follow_up_window")

    def close_follow_up_window(self) -> None:
        self._follow_up_deadline = 0.0
        if not self.is_muted:
            self._emit_state("listening")

    def _emit_state(self, state: str) -> None:
        if self._on_state_change is not None:
            try:
                self._on_state_change(state)
            except Exception:
                pass

    def evaluate_utterance(
        self,
        audio_pcm16: np.ndarray,
        in_follow_up: bool = False,
    ) -> GateEvaluation:
        """Run strict biometric verification BEFORE transcription.

        If similarity < threshold (0.78), returns accepted=False and transcription=None
        without ever invoking faster-whisper or system execution.
        """
        bio_res = self.biometrics.verify(audio_pcm16, sample_rate=self.config.audio.sample_rate)
        if not bio_res.verified:
            logger.info(
                "Biometric Gate REJECTED audio (sim=%.4f < %.2f, reason=%s, follow_up=%s)",
                bio_res.similarity,
                bio_res.threshold,
                bio_res.reason,
                in_follow_up,
            )
            return GateEvaluation(
                accepted=False,
                biometric=bio_res,
                transcription=None,
                in_follow_up_window=in_follow_up,
            )

        logger.info(
            "Biometric Gate VERIFIED owner (sim=%.4f >= %.2f, latency=%.1f ms)",
            bio_res.similarity,
            bio_res.threshold,
            bio_res.latency_ms,
        )
        stt_res = self.transcriber.transcribe(
            audio_pcm16, sample_rate=self.config.audio.sample_rate
        )
        return GateEvaluation(
            accepted=True,
            biometric=bio_res,
            transcription=stt_res,
            in_follow_up_window=in_follow_up,
        )

    def _capture_command_from_stream(self, stream) -> np.ndarray:
        """Capture a complete spoken utterance after wake trigger or in follow-up window."""
        sr = self.config.audio.sample_rate
        chunk_samples = self.config.audio.chunk_samples
        pre_roll = self.ring_buffer.get_last_samples(
            int(sr * self.config.audio.pre_roll_seconds)
        )

        captured: list[np.ndarray] = [pre_roll] if pre_roll.size > 0 else []
        max_chunks = max(1, int((self.config.audio.max_command_seconds * sr) / chunk_samples))
        silence_limit = max(
            1, int((self.config.audio.silence_timeout_seconds * sr) / chunk_samples)
        )
        heard_speech = False
        silent_chunks = 0

        for _ in range(max_chunks):
            if self._stop_event.is_set() or self._muted.is_set():
                break
            raw = stream.read(chunk_samples, exception_on_overflow=False)
            frame = np.frombuffer(raw, dtype=np.int16).copy()
            if self.vad.is_speech(frame):
                heard_speech = True
                silent_chunks = 0
                captured.append(frame)
            else:
                if heard_speech:
                    captured.append(frame)
                    silent_chunks += 1
                    if silent_chunks >= silence_limit:
                        break
                else:
                    # Wait briefly for user to start speaking after wake word
                    silent_chunks += 1
                    if silent_chunks >= int((3.0 * sr) / chunk_samples):
                        break

        if not heard_speech or not captured:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(captured)

    def capture_confirmation_utterance(self, timeout_seconds: float = 12.0) -> GateEvaluation | None:
        """Capture and biometrically verify a spoken confirmation reply (e.g. 'Yes, execute')."""
        if self._stream is None:
            return None
        sr = self.config.audio.sample_rate
        chunk_samples = self.config.audio.chunk_samples
        max_wait_chunks = max(1, int((timeout_seconds * sr) / chunk_samples))
        silence_limit = max(1, int((0.9 * sr) / chunk_samples))

        captured: list[np.ndarray] = []
        heard_speech = False
        silent_chunks = 0

        for _ in range(max_wait_chunks):
            if self._stop_event.is_set():
                return None
            if self._tts_speaking.is_set():
                time.sleep(0.05)
                continue
            raw = self._stream.read(chunk_samples, exception_on_overflow=False)
            frame = np.frombuffer(raw, dtype=np.int16).copy()
            if self.vad.is_speech(frame):
                heard_speech = True
                silent_chunks = 0
                captured.append(frame)
            elif heard_speech:
                captured.append(frame)
                silent_chunks += 1
                if silent_chunks >= silence_limit:
                    break

        if not heard_speech or not captured:
            return None
        audio = np.concatenate(captured)
        return self.evaluate_utterance(audio, in_follow_up=True)

    def run_forever(self) -> None:
        """Start the persistent microphone loop (blocking until stop() is called)."""
        import pyaudio

        if self.wake_detector is None:
            self.wake_detector = WakeWordDetector(config=self.config)

        if not self.biometrics.is_enrolled:
            logger.warning(
                "No enrolled voiceprint found at %s. All commands will be rejected until "
                "`python enroll_voice.py` is completed.",
                self.biometrics.voiceprint_path,
            )

        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self.config.audio.channels,
            rate=self.config.audio.sample_rate,
            input=True,
            input_device_index=self.config.audio.input_device_index,
            frames_per_buffer=self.config.audio.chunk_samples,
        )
        self._emit_state("listening")
        logger.info("AudioSecurityGate listening on microphone...")

        try:
            while not self._stop_event.is_set():
                raw = self._stream.read(
                    self.config.audio.chunk_samples, exception_on_overflow=False
                )
                if self._muted.is_set() or self._tts_speaking.is_set():
                    continue

                frame = np.frombuffer(raw, dtype=np.int16).copy()
                self.ring_buffer.append(frame)

                # Case A: Inside 10s continuous conversation window (no wake word required,
                # but speaker biometric verification is STILL enforced!)
                if self.in_follow_up_window:
                    if self.vad.is_speech(frame):
                        self._emit_state("capturing")
                        utt = self._capture_command_from_stream(self._stream)
                        if utt.size > 0:
                            self._handle_captured_utterance(utt, in_follow_up=True)
                        if not self.in_follow_up_window:
                            self._emit_state("listening")
                    continue

                # Case B: Idle state — monitor lightweight wake-word detector
                wake = self.wake_detector.feed(frame)
                if wake.detected:
                    logger.info("Wake word '%s' triggered (score=%.3f).", wake.keyword, wake.score)
                    self._emit_state("wake_triggered")
                    utt = self._capture_command_from_stream(self._stream)
                    if utt.size > 0:
                        self._handle_captured_utterance(utt, in_follow_up=False)
                    if not self.in_follow_up_window and not self._muted.is_set():
                        self._emit_state("listening")
        finally:
            self.release_audio()

    def _handle_captured_utterance(self, audio_pcm16: np.ndarray, in_follow_up: bool) -> None:
        self._emit_state("verifying")
        evaluation = self.evaluate_utterance(audio_pcm16, in_follow_up=in_follow_up)
        if not evaluation.accepted:
            # PRD Core Security Mandate: Discard audio and remain silent.
            if not self.in_follow_up_window:
                self._emit_state("listening")
            return

        if evaluation.transcription and evaluation.transcription.text:
            self._emit_state("processing")
            if self._on_command is not None:
                self._on_command(evaluation)
            # Keep 10-second continuous follow-up window active after verified command
            self.extend_follow_up_window()
        else:
            if not self.in_follow_up_window:
                self._emit_state("listening")

    def release_audio(self) -> None:
        """Immediately release PyAudio stream and hardware microphone access."""
        stream = self._stream
        pa = self._pa
        self._stream = None
        self._pa = None
        if stream is not None:
            try:
                if stream.is_active():
                    stream.stop_stream()
                stream.close()
            except Exception:
                pass
        if pa is not None:
            try:
                pa.terminate()
            except Exception:
                pass
        logger.info("Microphone stream released.")

    def stop(self) -> None:
        """Signal the audio loop to stop and release microphone immediately."""
        self._stop_event.set()
        self.release_audio()
