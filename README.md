# Project Jarvis — Voice-Only macOS System Assistant

Local-first, always-listening macOS desktop assistant protected by **Speaker Biometric Verification** (`>= 0.78` cosine similarity). Unrecognized voices or background audio are discarded silently at the audio pipeline layer before reaching speech-to-text or system execution.

## Architecture

- **Layer 1 (`core/wake_word.py`, `core/audio_gate.py`)**: Lightweight rolling ring buffer + `openWakeWord` (`hey_jarvis`) / optional `pvporcupine` + WebRTC VAD speech capture.
- **Layer 2 (`core/biometrics.py`, `enroll_voice.py`)**: Speaker embedding extraction (`SpeechBrain` ECAPA-TDNN or `Resemblyzer` GE2E) gated against `owner_voiceprint.npy` (`cosine_similarity >= 0.78`), plus 10-second continuous conversation window (FR-1).
- **Layer 3 (`core/stt_tts.py`, `core/orchestrator.py`)**: Local `faster-whisper` transcription (invoked only after biometric verification) + fast-path tool router + Gemini / Ollama tool-calling + macOS `say` / Kokoro TTS.
- **Layer 4 (`tools/mac_tools.py`, `core/safety.py`, `core/hotkey.py`)**: `NSWorkspace` / `open -a` app lifecycle, OS settings, Spotlight `mdfind`, `AXUIElement` UI automation, `yt-dlp` YouTube search & Chrome/Safari JS playback controls, spoken `"Yes, execute"` destructive confirmation, and `Cmd+Shift+Escape` emergency kill-switch.
- **Daemon (`core/daemon.py`, `jarvis.py`, `launchd/com.user.jarvis.plist`)**: macOS menu bar app (`rumps`) and `LaunchAgent` background service.

## Setup & Usage

1. **Audit macOS Permissions** (Microphone, Accessibility, Automation, Full Disk Access):
   ```zsh
   ./.venv/bin/python scripts/check_permissions.py
   ```
2. **Enroll Your Voiceprint** (5–10 calibration phrases):
   ```zsh
   ./.venv/bin/python enroll_voice.py --phrases 7 --verify
   ```
3. **Run Health Check**:
   ```zsh
   ./.venv/bin/python jarvis.py --check
   ```
4. **Launch Jarvis** (Menu Bar mode or `--headless`):
   ```zsh
   ./.venv/bin/python jarvis.py
   ```
   Optional LLM environment variables:
   - `GEMINI_API_KEY` for Gemini tool-calling
   - `OLLAMA_MODEL` / local Ollama server at `http://localhost:11434`

5. **Install Background LaunchAgent** (auto-start at login):
   ```zsh
   ./scripts/install_launchd.sh
   ```

6. **Run Automated Tests**:
   ```zsh
   ./.venv/bin/pytest -v
   ```
