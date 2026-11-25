"""Browser Audio Streaming, Adaptive VAD Gate, Speaker Biometrics, STT, and TTS."""

from __future__ import annotations

import base64
import collections
import logging
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional

import numpy as np

from config.settings import JarvisConfig, load_config
from core.biometrics import (
    SPEAKER_THRESHOLD,
    BiometricResult,
    VoiceBiometricGate,
    compute_rms_dbfs,
    normalize_audio_f32,
    pcm16_to_float32,
    resample_to_16k_f32,
)
from core.stt_tts import SpeechTranscriber
from tts_jarvis import JarvisTTS, SynthesizedAudio

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VoiceAuthResult:
    authorized: bool
    similarity: float
    threshold: float
    reason: str
    text: str
    biometric_ms: float
    stt_ms: float


class WebAudioSegmenter:
    """Adaptive VAD segmenter enforcing >= 1.2s continuous speech & 400ms hangover.

    Short clicks, breaths, or ambient bursts (< 1.2s continuous speech above the
    adaptive RMS floor) are discarded silently without triggering speaker verification.
    """

    def __init__(
        self,
        config: JarvisConfig,
        min_continuous_override: Optional[float] = None,
    ) -> None:
        self.cfg = config.audio
        self.sample_rate = self.cfg.sample_rate
        min_cont_sec = (
            min_continuous_override
            if min_continuous_override is not None
            else self.cfg.min_continuous_speech_seconds
        )
        self.min_continuous_samples = int(min_cont_sec * self.sample_rate)
        self.hangover_samples = int(
            self.cfg.trailing_hangover_seconds * self.sample_rate
        )
        self.silence_limit_samples = int(
            max(self.cfg.silence_timeout_seconds, self.cfg.trailing_hangover_seconds)
            * self.sample_rate
        )
        self.max_samples = int(self.cfg.max_command_seconds * self.sample_rate)

        self._vad = None
        try:
            import webrtcvad

            self._vad = webrtcvad.Vad(self.cfg.vad_aggressiveness)
        except Exception as exc:
            logger.warning("webrtcvad unavailable (%s); using adaptive RMS VAD.", exc)

        self._ambient_dbfs: float = float(self.cfg.energy_threshold_dbfs) - 6.0
        self._pre_roll: Deque[np.ndarray] = collections.deque(
            maxlen=max(1, int((self.cfg.pre_roll_seconds * self.sample_rate) / 512))
        )
        self._speech_frames: list[np.ndarray] = []
        self._in_speech = False
        self._silent_samples = 0
        self._speech_samples = 0
        self._continuous_voiced_samples = 0
        self._max_continuous_voiced_samples = 0
        self._trailing_silent_chunks: list[np.ndarray] = []

    def reset(self) -> None:
        self._pre_roll.clear()
        self._speech_frames.clear()
        self._trailing_silent_chunks.clear()
        self._in_speech = False
        self._silent_samples = 0
        self._speech_samples = 0
        self._continuous_voiced_samples = 0
        self._max_continuous_voiced_samples = 0

    @property
    def adaptive_threshold_dbfs(self) -> float:
        """Dynamic energy threshold sitting 9 dB above ambient noise floor."""
        return float(
            max(self.cfg.energy_threshold_dbfs, min(-26.0, self._ambient_dbfs + 9.0))
        )

    def _is_chunk_speech(self, pcm16: np.ndarray) -> bool:
        f32 = pcm16_to_float32(pcm16)
        rms_db = compute_rms_dbfs(f32)
        thresh = self.adaptive_threshold_dbfs

        if rms_db < thresh:
            # Update ambient noise floor slowly on quiet frames
            if rms_db > -95.0:
                self._ambient_dbfs = 0.94 * self._ambient_dbfs + 0.06 * rms_db
            return False

        if self._vad is None:
            return True

        sub_len = int(self.sample_rate * 0.02)  # 20 ms
        hits = 0
        total = 0
        for start in range(0, pcm16.size - sub_len + 1, sub_len):
            total += 1
            if self._vad.is_speech(pcm16[start : start + sub_len].tobytes(), self.sample_rate):
                hits += 1

        if total == 0:
            return True
        is_sp = hits >= max(1, total // 3)
        if not is_sp and rms_db < -28.0:
            self._ambient_dbfs = 0.96 * self._ambient_dbfs + 0.04 * rms_db
        return is_sp

    def feed(self, pcm16: np.ndarray, sample_rate: int = 16000) -> Optional[np.ndarray]:
        """Consume an incoming audio chunk; returns utterance only if >= 1.2s speech."""
        if pcm16.size == 0:
            return None

        if sample_rate != self.sample_rate:
            f32 = resample_to_16k_f32(pcm16_to_float32(pcm16), sample_rate=sample_rate)
            pcm16 = (np.clip(f32, -1.0, 1.0) * 32767.0).astype(np.int16)

        is_voiced = self._is_chunk_speech(pcm16)

        if not self._in_speech:
            self._pre_roll.append(pcm16)
            if is_voiced:
                self._in_speech = True
                self._speech_frames = list(self._pre_roll)
                self._speech_samples = sum(f.size for f in self._speech_frames)
                self._continuous_voiced_samples = pcm16.size
                self._max_continuous_voiced_samples = pcm16.size
                self._silent_samples = 0
                self._trailing_silent_chunks.clear()
            return None

        self._speech_samples += pcm16.size

        if is_voiced:
            # Re-attach any mid-sentence silence chunks before appending new speech
            if self._trailing_silent_chunks:
                self._speech_frames.extend(self._trailing_silent_chunks)
                self._trailing_silent_chunks.clear()
            self._speech_frames.append(pcm16)
            self._silent_samples = 0
            self._continuous_voiced_samples += pcm16.size
            if self._continuous_voiced_samples > self._max_continuous_voiced_samples:
                self._max_continuous_voiced_samples = self._continuous_voiced_samples
        else:
            self._silent_samples += pcm16.size
            self._trailing_silent_chunks.append(pcm16)
            # Small intra-word gaps (< 200ms) do not break continuous speech run
            if self._silent_samples > int(0.20 * self.sample_rate):
                self._continuous_voiced_samples = 0

        if self._silent_samples >= self.silence_limit_samples or self._speech_samples >= self.max_samples:
            # Keep exactly 400ms trailing hangover buffer after last voiced frame
            kept_tail: list[np.ndarray] = []
            tail_samples = 0
            for chunk in self._trailing_silent_chunks:
                if tail_samples >= self.hangover_samples:
                    break
                kept_tail.append(chunk)
                tail_samples += chunk.size

            utt = np.concatenate(self._speech_frames + kept_tail) if self._speech_frames else np.zeros(0, dtype=np.int16)
            max_run = self._max_continuous_voiced_samples
            self.reset()

            # Discard short clicks, breaths, or noise under 1.2s silently
            if max_run >= self.min_continuous_samples:
                return utt
            logger.debug(
                "Silently discarded transient audio (%.2fs < %.2fs required).",
                max_run / float(self.sample_rate),
                self.cfg.min_continuous_speech_seconds,
            )
        return None

    def flush(self) -> Optional[np.ndarray]:
        """Force-emit buffered utterance if it met the minimum speech requirement."""
        if not self._in_speech or not self._speech_frames:
            self.reset()
            return None
        utt = np.concatenate(self._speech_frames + self._trailing_silent_chunks)
        max_run = self._max_continuous_voiced_samples
        self.reset()
        return utt if max_run >= self.min_continuous_samples else None


class BrowserEnrollmentSession:
    """Collects 3-5 normalized calibration clips from /enroll to build centroid voiceprint."""

    PHRASES = [
        "Jarvis, open Visual Studio Code and check my active workspace.",
        "Jarvis, search and play lo-fi beats on YouTube.",
        "Jarvis, set system volume to fifty percent and enable dark mode.",
        "Jarvis, find all PDF documents in my Downloads folder.",
        "Jarvis, pause playback and confirm yes execute.",
    ]

    def __init__(
        self,
        gate: VoiceBiometricGate,
        min_phrases: int = 3,
        max_phrases: int = 5,
    ) -> None:
        self.gate = gate
        self.min_phrases = min_phrases
        self.max_phrases = max_phrases
        self.clips: dict[int, np.ndarray] = {}

    def add_phrase(self, index: int, pcm16: np.ndarray, sample_rate: int = 16000) -> dict:
        if not (0 <= index < self.max_phrases):
            raise ValueError(f"Phrase index must be 0..{self.max_phrases - 1}")

        f32 = resample_to_16k_f32(pcm16_to_float32(pcm16), sample_rate=sample_rate)
        dur = f32.size / 16000.0
        rms = compute_rms_dbfs(f32)
        if dur < 1.5:
            raise ValueError("Sample too short (< 1.5s). Please record 3–5 seconds of clear speech.")
        if rms < -52.0:
            raise ValueError("Audio too quiet or silent. Check your browser microphone.")

        norm_f32 = normalize_audio_f32(f32)
        norm_pcm16 = (np.clip(norm_f32, -1.0, 1.0) * 32767.0).astype(np.int16)
        self.clips[index] = norm_pcm16
        return {
            "index": index,
            "rms_dbfs": round(rms, 1),
            "duration_sec": round(dur, 2),
            "recorded_count": len(self.clips),
            "min_phrases": self.min_phrases,
            "max_phrases": self.max_phrases,
        }

    def finalize(self, sample_rate: int = 16000) -> dict:
        if len(self.clips) < self.min_phrases:
            raise ValueError(
                f"Need at least {self.min_phrases} calibration samples, got {len(self.clips)}."
            )
        ordered = [self.clips[i] for i in sorted(self.clips.keys())]
        centroid, sims, mean_sim = self.gate.enroll_from_clips(ordered, sample_rate=16000)
        saved_path = self.gate.save_voiceprint(
            centroid,
            phrase_count=len(ordered),
            mean_consistency=mean_sim,
        )
        return {
            "voiceprint_path": str(saved_path),
            "phrase_count": len(ordered),
            "phrase_similarities": [round(float(s), 4) for s in sims],
            "mean_consistency": round(float(mean_sim), 4),
            "threshold": float(self.gate.threshold),
            "engine": self.gate.backend.name,
        }


class WebVoiceAuthService:
    """Coordinates audio conditioning, faster-whisper STT, and British J.A.R.V.I.S. TTS."""

    def __init__(self, config: Optional[JarvisConfig] = None) -> None:
        self.config = config or load_config()
        self.transcriber = SpeechTranscriber(config=self.config)
        self.tts = JarvisTTS(config=self.config)

    def warmup(self) -> None:
        """Pre-load Whisper STT for instant readiness."""
        try:
            self.transcriber.warmup()
        except Exception as exc:
            logger.warning("Whisper STT warmup warning: %s", exc)

    def transcribe_utterance(
        self,
        pcm16: np.ndarray,
        sample_rate: int = 16000,
    ) -> VoiceAuthResult:
        """Condition incoming browser audio and run faster-whisper STT directly."""
        f32 = resample_to_16k_f32(pcm16_to_float32(pcm16), sample_rate=sample_rate)
        cond_f32 = normalize_audio_f32(f32)
        cond_pcm16 = (np.clip(cond_f32, -1.0, 1.0) * 32767.0).astype(np.int16)

        stt = self.transcriber.transcribe(cond_pcm16, sample_rate=16000)
        return VoiceAuthResult(
            authorized=True,
            similarity=1.0,
            threshold=0.0,
            reason="open_access",
            text=stt.text,
            biometric_ms=0.0,
            stt_ms=stt.latency_ms,
        )

    def verify_and_transcribe(
        self,
        pcm16: np.ndarray,
        sample_rate: int = 16000,
    ) -> VoiceAuthResult:
        """Direct transcription entrypoint (voice-lock removed)."""
        return self.transcribe_utterance(pcm16, sample_rate=sample_rate)

    async def synthesize_tts_async(self, text: str) -> Optional[SynthesizedAudio]:
        """Generate British J.A.R.V.I.S. voice audio via edge-tts / offline fallback."""
        return await self.tts.synthesize_async(text)

    def synthesize_tts_wav_b64(self, text: str) -> Optional[str]:
        """Backwards-compatible sync helper returning base64 audio string."""
        res = self.tts.synthesize_sync(text)
        return res.audio_b64 if res else None
