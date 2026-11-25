#!/usr/bin/env python3
"""Main entrypoint for Project Jarvis macOS Voice-Only System Assistant."""

from __future__ import annotations

import argparse
import sys

from config.settings import load_config
from core.daemon import configure_logging, JarvisDaemon, run_menu_bar_app


def main() -> int:
    parser = argparse.ArgumentParser(description="Project Jarvis — Voice-Only macOS System Assistant")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without the macOS menu bar status icon.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify configuration, enrolled voiceprint, and module readiness, then exit.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to custom settings.yaml.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_logging(cfg)

    if args.check:
        from core.biometrics import VoiceBiometricGate

        gate = VoiceBiometricGate(config=cfg)
        print(f"Project Root : {cfg.project_root}")
        print(f"Biometrics   : {cfg.biometrics.engine} (threshold >= {gate.threshold:.2f})")
        print(f"Voiceprint   : {gate.voiceprint_path} (enrolled={gate.is_enrolled})")
        print(f"Wake Word    : {cfg.wake_word.engine} ({cfg.wake_word.model_name})")
        print(f"STT / TTS    : {cfg.stt.engine} ({cfg.stt.model_size}) / {cfg.tts.engine}")
        print(f"Kill-Switch  : {cfg.safety.kill_switch_hotkey}")
        return 0

    if args.headless or not cfg.daemon.menu_bar_enabled:
        JarvisDaemon(config=cfg).run_headless()
    else:
        run_menu_bar_app(config=cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
