"""Iron Man J.A.R.V.I.S. British Butler Voice Synthesis & Boot Chime Module.

Primary engine: `edge-tts` using `en-GB-RyanNeural` (rate="+4%", pitch="-3Hz").
Offline fallbacks: `Kokoro-TTS` (`bm_george` / `bm_lewis`) -> macOS `say -v Daniel`.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import subprocess
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from config.settings import JarvisConfig, load_config

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
CHIME_PATH = PROJECT_ROOT / "static" / "chime.wav"


@dataclass(frozen=True)
class SynthesizedAudio:
    audio_b64: str
    mime_type: str
    raw_bytes: bytes
    engine: str


def ensure_boot_chime_wav(path: Path = CHIME_PATH) -> Path:
    """Generate a subtle, crisp harmonic boot chime WAV if not already cached."""
    if path.exists() and path.stat().st_size > 1000:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    sr = 22050
    dur = 0.42
    t = np.linspace(0.0, dur, int(sr * dur), endpoint=False, dtype=np.float64)

    # Clean two-stage ascending shimmer: C5 (523.25Hz) -> G5 (783.99Hz) + C6 (1046.5Hz)
    env1 = np.exp(-t * 9.0) * (1.0 - np.exp(-t * 140.0))
    t2 = np.maximum(0.0, t - 0.08)
    env2 = np.where(t >= 0.08, np.exp(-t2 * 7.5) * (1.0 - np.exp(-t2 * 140.0)), 0.0)

    sig = (
        0.35 * env1 * np.sin(2.0 * np.pi * 523.25 * t)
        + 0.30 * env2 * np.sin(2.0 * np.pi * 783.99 * t)
        + 0.20 * env2 * np.sin(2.0 * np.pi * 1046.50 * t)
    )
    pcm16 = (np.clip(sig, -0.95, 0.95) * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())
    return path


class JarvisTTS:
    """British J.A.R.V.I.S. speech synthesizer with TTS_PLAYING mic-suppression state."""

    def __init__(
        self,
        config: Optional[JarvisConfig] = None,
        on_speaking_change: Optional[Callable[[bool], None]] = None,
    ) -> None:
        self.config = config or load_config()
        self._on_speaking_change = on_speaking_change
        self._speaking_lock = threading.Lock()
        self.is_speaking = False
        self._kokoro_pipeline = None
        ensure_boot_chime_wav()

    def set_speaking_callback(self, cb: Optional[Callable[[bool], None]]) -> None:
        self._on_speaking_change = cb

    def _set_speaking(self, speaking: bool) -> None:
        self.is_speaking = speaking
        if self._on_speaking_change is not None:
            try:
                self._on_speaking_change(speaking)
            except Exception:
                pass

    async def synthesize_async(self, text: str) -> Optional[SynthesizedAudio]:
        """Synthesize text to MP3 (edge-tts) or WAV (Kokoro / macOS Daniel fallback)."""
        cleaned = (text or "").strip()
        if not cleaned:
            return None

        engine = self.config.tts.engine.lower()
        if engine == "edge-tts":
            try:
                res = await self._synth_edge_tts(cleaned)
                if res is not None:
                    return res
            except Exception as exc:
                logger.warning("edge-tts failed (%s); falling back to macOS Daniel.", exc)

        if engine in ("edge-tts", "kokoro"):
            try:
                res = await asyncio.to_thread(self._synth_kokoro_wav, cleaned)
                if res is not None:
                    return res
            except Exception as exc:
                logger.warning("Kokoro TTS failed (%s); falling back to macOS Daniel.", exc)

        return await asyncio.to_thread(self._synth_macos_daniel_wav, cleaned)

    def synthesize_sync(self, text: str) -> Optional[SynthesizedAudio]:
        """Synchronous wrapper around synthesize_async for thread contexts."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Run in dedicated worker thread with its own event loop
            result_box: list[Optional[SynthesizedAudio]] = []
            err_box: list[Exception] = []

            def _runner():
                try:
                    result_box.append(asyncio.run(self.synthesize_async(text)))
                except Exception as exc:
                    err_box.append(exc)

            t = threading.Thread(target=_runner, daemon=True)
            t.start()
            t.join(timeout=15.0)
            if err_box:
                logger.warning("TTS sync synthesis error: %s", err_box[0])
            return result_box[0] if result_box else None

        return asyncio.run(self.synthesize_async(text))

    async def _synth_edge_tts(self, text: str) -> Optional[SynthesizedAudio]:
        try:
            import edge_tts

            voice = self.config.tts.edge_voice or "en-GB-RyanNeural"
            rate = self.config.tts.edge_rate or "+4%"
            pitch = self.config.tts.edge_pitch or "-3Hz"
            comm = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)

            buf = bytearray()
            async for chunk in comm.stream():
                if chunk.get("type") == "audio":
                    buf.extend(chunk["data"])
            if not buf:
                return None
            raw = bytes(buf)
            return SynthesizedAudio(
                audio_b64=base64.b64encode(raw).decode("ascii"),
                mime_type="audio/mpeg",
                raw_bytes=raw,
                engine=f"edge-tts:{voice}",
            )
        except Exception as exc:
            logger.warning("edge-tts unavailable (%s); falling back offline.", exc)
            return None

    def _synth_kokoro_wav(self, text: str) -> Optional[SynthesizedAudio]:
        try:
            if self._kokoro_pipeline is None:
                from kokoro import KPipeline  # type: ignore

                self._kokoro_pipeline = KPipeline(lang_code=self.config.tts.kokoro_lang or "b")

            chunks: list[np.ndarray] = []
            voice = self.config.tts.kokoro_voice or "bm_george"
            for _, _, audio in self._kokoro_pipeline(text, voice=voice):
                chunks.append(np.asarray(audio, dtype=np.float32))
            if not chunks:
                return None

            merged = np.concatenate(chunks)
            pcm16 = (np.clip(merged, -1.0, 1.0) * 32767.0).astype(np.int16)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(pcm16.tobytes())
            raw = buf.getvalue()
            return SynthesizedAudio(
                audio_b64=base64.b64encode(raw).decode("ascii"),
                mime_type="audio/wav",
                raw_bytes=raw,
                engine=f"kokoro:{voice}",
            )
        except Exception:
            return None

    def _synth_macos_daniel_wav(self, text: str) -> Optional[SynthesizedAudio]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = Path(tmp.name)
        try:
            voice = self.config.tts.macos_voice or "Daniel"
            rate = str(self.config.tts.macos_rate or 195)
            proc = subprocess.run(
                [
                    "say",
                    "-v",
                    voice,
                    "-r",
                    rate,
                    "--data-format=LEI16@22050",
                    "-o",
                    str(wav_path),
                    text,
                ],
                capture_output=True,
                check=False,
                timeout=15.0,
            )
            if proc.returncode != 0 or not wav_path.exists():
                return None
            raw = wav_path.read_bytes()
            return SynthesizedAudio(
                audio_b64=base64.b64encode(raw).decode("ascii"),
                mime_type="audio/wav",
                raw_bytes=raw,
                engine=f"macos-say:{voice}",
            )
        except Exception as exc:
            logger.warning("macOS say fallback failed: %s", exc)
            return None
        finally:
            wav_path.unlink(missing_ok=True)

    def play_boot_chime_local(self) -> None:
        """Play the sci-fi boot chime through macOS speakers."""
        chime = ensure_boot_chime_wav()
        with self._speaking_lock:
            self._set_speaking(True)
            try:
                subprocess.run(
                    ["afplay", str(chime)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=5.0,
                )
            finally:
                time.sleep(0.15)
                self._set_speaking(False)

    def speak_local(self, text: str) -> None:
        """Synthesize and play speech through macOS speakers while holding TTS_PLAYING."""
        synth = self.synthesize_sync(text)
        if synth is None:
            return
        suffix = ".mp3" if synth.mime_type == "audio/mpeg" else ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)
        tmp_path.write_bytes(synth.raw_bytes)
        with self._speaking_lock:
            self._set_speaking(True)
            try:
                subprocess.run(
                    ["afplay", str(tmp_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=25.0,
                )
            finally:
                time.sleep(0.35)
                self._set_speaking(False)
                tmp_path.unlink(missing_ok=True)
