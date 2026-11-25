"""Strongly-typed configuration loader for Project Jarvis."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


class AudioConfig(BaseModel):
    sample_rate: int = 16000
    channels: int = 1
    chunk_samples: int = 1280
    ring_buffer_seconds: float = 3.0
    pre_roll_seconds: float = 0.5
    max_command_seconds: float = 12.0
    silence_timeout_seconds: float = 0.85
    trailing_hangover_seconds: float = 0.40
    min_speech_seconds: float = 0.35
    min_continuous_speech_seconds: float = 1.20
    vad_aggressiveness: int = Field(default=2, ge=0, le=3)
    energy_threshold_dbfs: float = -42.0
    input_device_index: Optional[int] = None


class WakeWordConfig(BaseModel):
    engine: str = "openwakeword"
    model_name: str = "hey_jarvis"
    threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    patience_frames: int = Field(default=1, ge=1)
    cooldown_seconds: float = 1.5
    porcupine_keyword: str = "jarvis"
    porcupine_access_key_env: str = "PICOVOICE_ACCESS_KEY"


class BiometricsConfig(BaseModel):
    engine: str = "speechbrain"
    speechbrain_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    speechbrain_savedir: str = "config/models/spkrec-ecapa-voxceleb"
    similarity_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    voiceprint_path: str = "owner_voiceprint.npy"
    enrollment_phrases_count: int = Field(default=5, ge=3, le=10)
    min_phrase_consistency: float = Field(default=0.60, ge=0.0, le=1.0)
    strict_silence_on_reject: bool = True


class ConversationConfig(BaseModel):
    follow_up_window_seconds: float = Field(default=10.0, ge=1.0)


class STTConfig(BaseModel):
    engine: str = "faster-whisper"
    model_size: str = "small.en"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "en"
    beam_size: int = 1
    initial_prompt: str = "Jarvis, WhatsApp, YouTube, VS Code, Terminal, MacBook."


class TTSConfig(BaseModel):
    engine: str = "edge-tts"
    edge_voice: str = "en-GB-RyanNeural"
    edge_rate: str = "+4%"
    edge_pitch: str = "-3Hz"
    macos_voice: str = "Daniel"
    macos_rate: int = 195
    kokoro_voice: str = "bm_george"
    kokoro_lang: str = "b"


class LLMConfig(BaseModel):
    provider: str = "auto"
    gemini_model: str = "gemini-2.5-flash"
    gemini_api_key_env: str = "GEMINI_API_KEY"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_timeout_seconds: float = 15.0
    fast_path_enabled: bool = True


class SafetyConfig(BaseModel):
    confirm_destructive_actions: bool = True
    confirmation_phrase: str = "yes, execute"
    confirmation_timeout_seconds: float = 12.0
    kill_switch_hotkey: str = "cmd+shift+escape"
    audit_log_path: str = "logs/jarvis_audit.log"


class DaemonConfig(BaseModel):
    menu_bar_enabled: bool = True
    log_level: str = "INFO"
    log_file: str = "logs/jarvis.log"


class JarvisConfig(BaseModel):
    project_root: Path = PROJECT_ROOT
    audio: AudioConfig = Field(default_factory=AudioConfig)
    wake_word: WakeWordConfig = Field(default_factory=WakeWordConfig)
    biometrics: BiometricsConfig = Field(default_factory=BiometricsConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    daemon: DaemonConfig = Field(default_factory=DaemonConfig)

    def resolve_path(self, relative_or_abs: str | Path) -> Path:
        p = Path(relative_or_abs).expanduser()
        if p.is_absolute():
            return p
        return self.project_root / p


def _load_dotenv(env_path: Path) -> None:
    """Load KEY=VALUE pairs from .env without overriding already-exported env vars."""
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, val = stripped.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def load_config(config_path: Optional[str | Path] = None) -> JarvisConfig:
    """Load JarvisConfig from YAML file with optional .env and environment variable overrides."""
    _load_dotenv(PROJECT_ROOT / ".env")
    path = Path(config_path) if config_path else Path(os.environ.get("JARVIS_CONFIG", DEFAULT_CONFIG_PATH))
    raw: dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    cfg = JarvisConfig(**raw)

    # Environment overrides for common runtime knobs
    if env_thresh := os.environ.get("JARVIS_BIOMETRIC_THRESHOLD"):
        cfg.biometrics.similarity_threshold = float(env_thresh)
    if env_engine := os.environ.get("JARVIS_BIOMETRIC_ENGINE"):
        cfg.biometrics.engine = env_engine.strip().lower()
    if env_vp := os.environ.get("JARVIS_VOICEPRINT_PATH"):
        cfg.biometrics.voiceprint_path = env_vp
    if env_llm := os.environ.get("JARVIS_LLM_PROVIDER"):
        cfg.llm.provider = env_llm.strip().lower()

    return cfg
