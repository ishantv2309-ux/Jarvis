"""Low-CPU Background Standby Listener for 'Jarvis wake up' & macOS Browser Focus."""

from __future__ import annotations

import logging
import re
import subprocess
import threading
import time
from typing import Callable, Optional

import numpy as np

from config.settings import JarvisConfig, load_config
from core.biometrics import compute_rms_dbfs, pcm16_to_float32
from core.stt_tts import SpeechTranscriber
from core.wake_word import SpeechActivityDetector

logger = logging.getLogger(__name__)

_WAKE_PATTERNS = (
    re.compile(r"\bjarvis\s+wake\s*up\b", re.IGNORECASE),
    re.compile(r"\bwake\s*up\s+jarvis\b", re.IGNORECASE),
    re.compile(r"\bhey\s+jarvis\b", re.IGNORECASE),
    re.compile(r"\bjarvis\b", re.IGNORECASE),
)

_WAKE_EXTRACT_RE = re.compile(
    r"^(?:.*?\b(?:wake\s*up\s+jarvis|jarvis\s+wake\s*up|hey\s+jarvis|ok\s+jarvis|jarvis)\b[\s,.:;!?-]*)(.*)$",
    re.IGNORECASE,
)


def extract_jarvis_command(text: str) -> tuple[bool, str]:
    """Check if transcript contains a wake call ('jarvis', 'hey jarvis', 'wake up jarvis').

    Returns:
        (wake_called, remaining_task_text)
    """
    raw = (text or "").strip()
    if not raw:
        return False, ""
    if not matches_wake_phrase(raw):
        return False, ""
    m = _WAKE_EXTRACT_RE.match(raw)
    if not m:
        return True, ""
    task = m.group(1).strip(" ,.:;!?-")
    return True, task

_SLEEP_PATTERNS = (
    re.compile(r"^(?:jarvis\s+)?go\s+to\s+sleep$", re.IGNORECASE),
    re.compile(r"^(?:jarvis\s+)?(?:enter\s+)?standby(?:\s+mode)?$", re.IGNORECASE),
    re.compile(r"^(?:jarvis\s+)?sleep\s+mode$", re.IGNORECASE),
)


def matches_wake_phrase(text: str) -> bool:
    """Return True if transcript contains 'Jarvis wake up', 'wake up Jarvis', or 'Hey Jarvis'."""
    norm = re.sub(r"[^\w\s]", " ", (text or "")).strip()
    norm = re.sub(r"\s+", " ", norm)
    return any(pat.search(norm) for pat in _WAKE_PATTERNS)


def matches_sleep_phrase(text: str) -> bool:
    """Return True if transcript asks Jarvis to enter standby / go to sleep."""
    norm = re.sub(r"[^\w\s]", " ", (text or "")).strip()
    norm = re.sub(r"\s+", " ", norm)
    return any(pat.match(norm) for pat in _SLEEP_PATTERNS)


def focus_or_open_jarvis_browser(url: str = "http://localhost:8000") -> None:
    """Bring an existing Jarvis browser tab to the foreground, or open it if closed."""
    applescript = f'''
set targetUrl to "{url}"
set foundTab to false

if application "Google Chrome" is running then
    tell application "Google Chrome"
        repeat with w in windows
            set tabIdx to 0
            repeat with t in tabs of w
                set tabIdx to tabIdx + 1
                if URL of t contains "localhost:8000" or URL of t contains "127.0.0.1:8000" then
                    set active tab index of w to tabIdx
                    set index of w to 1
                    activate
                    set foundTab to true
                    exit repeat
                end if
            end repeat
            if foundTab then exit repeat
        end repeat
    end tell
end if

if not foundTab and application "Safari" is running then
    tell application "Safari"
        repeat with w in windows
            repeat with t in tabs of w
                if URL of t contains "localhost:8000" or URL of t contains "127.0.0.1:8000" then
                    set current tab of w to t
                    set index of w to 1
                    activate
                    set foundTab to true
                    exit repeat
                end if
            end repeat
            if foundTab then exit repeat
        end repeat
    end tell
end if

if not foundTab then
    do shell script "open " & quoted form of targetUrl
end if
'''
    try:
        subprocess.run(
            ["osascript", "-e", applescript],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5.0,
        )
    except Exception as exc:
        logger.debug("focus_or_open_jarvis_browser fallback: %s", exc)
        subprocess.Popen(["open", url])


class StandbyWakeListener:
    """Low-CPU microphone listener active only in STANDBY mode.

    Uses WebRTC VAD + energy gating so Whisper runs ONLY on brief voiced bursts,
    listening exclusively for 'Jarvis wake up' / 'Wake up Jarvis' / 'Hey Jarvis'.
    """

    def __init__(
        self,
        config: Optional[JarvisConfig] = None,
        transcriber: Optional[SpeechTranscriber] = None,
        on_wake: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.config = config or load_config()
        self.transcriber = transcriber or SpeechTranscriber(config=self.config)
        self.on_wake = on_wake
        self.vad = SpeechActivityDetector(self.config.audio)

        self._stop_event = threading.Event()
        self._paused_event = threading.Event()
        self._tts_playing_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_wake_ts = 0.0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="jarvis-standby-wake",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def set_standby_active(self, standby_on: bool) -> None:
        """Pause background PyAudio capture when UI is ACTIVE (or resume in STANDBY)."""
        if standby_on:
            self._paused_event.clear()
        else:
            self._paused_event.set()

    def set_tts_playing(self, playing: bool) -> None:
        """Suppress mic ingestion while Jarvis is speaking (TTS_PLAYING guard)."""
        if playing:
            self._tts_playing_event.set()
        else:
            self._tts_playing_event.clear()

    def _run_loop(self) -> None:
        try:
            import pyaudio
        except Exception as exc:
            logger.warning("PyAudio unavailable for StandbyWakeListener: %s", exc)
            return

        sr = self.config.audio.sample_rate
        chunk_samples = self.config.audio.chunk_samples
        pa = None
        stream = None

        while not self._stop_event.is_set():
            if self._paused_event.is_set() or self._tts_playing_event.is_set():
                if stream is not None:
                    try:
                        stream.stop_stream()
                        stream.close()
                    except Exception:
                        pass
                    stream = None
                if pa is not None:
                    try:
                        pa.terminate()
                    except Exception:
                        pass
                    pa = None
                time.sleep(0.12)
                continue

            if stream is None:
                try:
                    pa = pyaudio.PyAudio()
                    stream = pa.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=sr,
                        input=True,
                        input_device_index=self.config.audio.input_device_index,
                        frames_per_buffer=chunk_samples,
                    )
                except Exception as exc:
                    logger.warning("Standby mic open failed (%s); retrying in 2s.", exc)
                    time.sleep(2.0)
                    continue

            try:
                raw = stream.read(chunk_samples, exception_on_overflow=False)
                frame = np.frombuffer(raw, dtype=np.int16).copy()
                if not self.vad.is_speech(frame):
                    continue

                # Capture short 0.5s - 3.2s phrase for wake keyword spotting
                frames = [frame]
                silent = 0
                max_chunks = int((3.2 * sr) / chunk_samples)
                sil_limit = int((0.55 * sr) / chunk_samples)
                for _ in range(max_chunks):
                    if (
                        self._stop_event.is_set()
                        or self._paused_event.is_set()
                        or self._tts_playing_event.is_set()
                    ):
                        break
                    nxt = np.frombuffer(
                        stream.read(chunk_samples, exception_on_overflow=False),
                        dtype=np.int16,
                    ).copy()
                    frames.append(nxt)
                    if self.vad.is_speech(nxt):
                        silent = 0
                    else:
                        silent += 1
                        if silent >= sil_limit:
                            break

                utt = np.concatenate(frames)
                if utt.size < int(0.55 * sr):
                    continue
                if compute_rms_dbfs(pcm16_to_float32(utt)) < -46.0:
                    continue

                res = self.transcriber.transcribe(utt, sample_rate=sr)
                if res.text and matches_wake_phrase(res.text):
                    now = time.monotonic()
                    if now - self._last_wake_ts >= 2.5:
                        self._last_wake_ts = now
                        logger.info("Standby wake phrase detected: %r", res.text)
                        if self.on_wake is not None:
                            self.on_wake(res.text)
            except Exception as exc:
                logger.debug("Standby loop warning: %s", exc)
                time.sleep(0.2)

        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        if pa is not None:
            try:
                pa.terminate()
            except Exception:
                pass
