"""Background Daemon & macOS Menu Bar Status App for Project Jarvis (Phase 4)."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

from config.settings import JarvisConfig, load_config
from core.audio_gate import AudioSecurityGate, GateEvaluation
from core.biometrics import VoiceBiometricGate
from core.hotkey import GlobalKillSwitch
from core.orchestrator import AgentOrchestrator
from core.safety import SafetyGuard
from core.stt_tts import SpeechSynthesizer, SpeechTranscriber
from tools.mac_tools import MacToolRegistry

logger = logging.getLogger(__name__)


def configure_logging(cfg: JarvisConfig) -> None:
    log_path = cfg.resolve_path(cfg.daemon.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, cfg.daemon.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )


class JarvisDaemon:
    """Coordinates audio security gate, orchestrator, TTS, safety, kill-switch, and UI."""

    def __init__(
        self,
        config: Optional[JarvisConfig] = None,
        on_state_change: Optional[callable] = None,
    ) -> None:
        self.config = config or load_config()
        self._external_state_cb = on_state_change
        self._shutting_down = threading.Event()

        self.biometrics = VoiceBiometricGate(config=self.config)
        self.transcriber = SpeechTranscriber(config=self.config)
        self.tts = SpeechSynthesizer(config=self.config)

        self.audio_gate = AudioSecurityGate(
            config=self.config,
            biometrics=self.biometrics,
            transcriber=self.transcriber,
            on_command=self._on_verified_command,
            on_state_change=self._handle_state_change,
        )
        self.tts.set_speaking_callback(self.audio_gate.set_tts_speaking)

        self.safety_guard = SafetyGuard(
            config=self.config,
            speak_fn=lambda msg: self.tts.speak(msg, block=True),
            listen_confirmation_fn=self._listen_for_confirmation,
        )
        self.tools = MacToolRegistry(safety_guard=self.safety_guard)
        self.orchestrator = AgentOrchestrator(config=self.config, tools=self.tools)
        self.kill_switch = GlobalKillSwitch(
            hotkey=self.config.safety.kill_switch_hotkey,
            on_trigger=self.emergency_shutdown,
        )
        self._audio_thread: Optional[threading.Thread] = None

    def _handle_state_change(self, state: str) -> None:
        if self._external_state_cb is not None:
            try:
                self._external_state_cb(state)
            except Exception:
                pass

    def _listen_for_confirmation(self, timeout_seconds: float) -> Optional[str]:
        eval_res = self.audio_gate.capture_confirmation_utterance(
            timeout_seconds=timeout_seconds
        )
        if eval_res is None or not eval_res.accepted or eval_res.transcription is None:
            return None
        return eval_res.transcription.text

    def _on_verified_command(self, evaluation: GateEvaluation) -> None:
        if not evaluation.transcription or not evaluation.transcription.text:
            return
        text = evaluation.transcription.text
        logger.info("Executing verified command: %r", text)
        response = self.orchestrator.handle_command(text)
        if response.spoken_text:
            self.tts.speak(response.spoken_text, block=True)

    def start_background_audio(self) -> None:
        """Warm up STT and start the microphone listener & kill-switch in daemon threads."""
        try:
            self.transcriber.warmup()
        except Exception as exc:
            logger.warning("STT warmup deferred (%s).", exc)

        self.kill_switch.start()
        self._audio_thread = threading.Thread(
            target=self.audio_gate.run_forever,
            daemon=True,
            name="jarvis-audio-gate",
        )
        self._audio_thread.start()

    def run_headless(self) -> None:
        """Run Jarvis without the macOS menu bar UI (e.g. CLI or background LaunchAgent)."""
        self.start_background_audio()
        signal.signal(signal.SIGINT, lambda *_: self.emergency_shutdown())
        signal.signal(signal.SIGTERM, lambda *_: self.emergency_shutdown())
        try:
            while not self._shutting_down.is_set():
                self._shutting_down.wait(timeout=0.5)
        finally:
            self.stop()

    def toggle_mute(self) -> bool:
        new_muted = not self.audio_gate.is_muted
        self.audio_gate.set_muted(new_muted)
        return new_muted

    def stop(self) -> None:
        if self._shutting_down.is_set():
            return
        self._shutting_down.set()
        self.audio_gate.stop()
        self.kill_switch.stop()
        logger.info("Jarvis daemon stopped cleanly.")

    def emergency_shutdown(self) -> None:
        """Immediate kill-switch termination releasing microphone access."""
        self.stop()
        try:
            import rumps

            rumps.quit_application()
        except Exception:
            pass
        os._exit(0)


def run_menu_bar_app(config: Optional[JarvisConfig] = None) -> None:
    """Launch Jarvis as a silent macOS menu bar daemon via rumps."""
    import rumps

    cfg = config or load_config()
    configure_logging(cfg)

    status_titles = {
        "listening": "Jarvis",
        "wake_triggered": "Jarvis*",
        "capturing": "Jarvis...",
        "verifying": "Jarvis?",
        "processing": "Jarvis!",
        "follow_up_window": "Jarvis+",
        "muted": "Jarvis (Muted)",
    }

    class JarvisMenuBarApp(rumps.App):
        def __init__(self) -> None:
            super().__init__("Jarvis", quit_button=None)
            self.state_item = rumps.MenuItem("Status: Starting...")
            self.voiceprint_item = rumps.MenuItem("Voiceprint: Checking...")
            self.mute_item = rumps.MenuItem("Mute Microphone", callback=self.on_toggle_mute)
            self.enroll_item = rumps.MenuItem("Enroll / Re-enroll Voice...", callback=self.on_enroll)
            self.quit_item = rumps.MenuItem(
                f"Quit Jarvis ({cfg.safety.kill_switch_hotkey})", callback=self.on_quit
            )
            self.menu = [
                self.state_item,
                self.voiceprint_item,
                None,
                self.mute_item,
                self.enroll_item,
                None,
                self.quit_item,
            ]
            self.daemon = JarvisDaemon(config=cfg, on_state_change=self.on_state_change)
            vp_status = (
                f"Enrolled ({cfg.biometrics.engine})"
                if self.daemon.biometrics.is_enrolled
                else "NOT ENROLLED (run enroll_voice.py)"
            )
            self.voiceprint_item.title = f"Voiceprint: {vp_status}"
            self.daemon.start_background_audio()

        def on_state_change(self, state: str) -> None:
            self.title = status_titles.get(state, "Jarvis")
            self.state_item.title = f"Status: {state.replace('_', ' ').title()}"

        def on_toggle_mute(self, sender) -> None:
            muted = self.daemon.toggle_mute()
            sender.title = "Unmute Microphone" if muted else "Mute Microphone"

        def on_enroll(self, _) -> None:
            enroll_script = Path(cfg.project_root) / "enroll_voice.py"
            subprocess.Popen(
                [
                    "osascript",
                    "-e",
                    f'tell application "Terminal" to do script "{sys.executable} {enroll_script} --verify"',
                ]
            )

        def on_quit(self, _) -> None:
            self.daemon.stop()
            rumps.quit_application()

    JarvisMenuBarApp().run()
