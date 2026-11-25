#!/usr/bin/env python3
"""Speaker Biometric Enrollment Utility (FR-1) for Project Jarvis.

Records 5-10 calibration phrases from the owner via the macOS microphone,
validates vocal consistency, generates the master voice embedding vector
`owner_voiceprint.npy`, and optionally verifies a live test phrase.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from config.settings import load_config
from core.biometrics import VoiceBiometricGate, compute_rms_dbfs, pcm16_to_float32

CALIBRATION_PHRASES = [
    "Jarvis, open Visual Studio Code and check my active workspace.",
    "Jarvis, search and play lo-fi beats on YouTube.",
    "Jarvis, set system volume to fifty percent and enable dark mode.",
    "Jarvis, find all PDF documents in my Downloads folder.",
    "Jarvis, pause playback and switch to Google Chrome.",
    "Jarvis, what applications are currently running on my Mac?",
    "Jarvis, turn on Do Not Disturb for the next hour.",
    "Jarvis, lower screen brightness and mute the microphone.",
    "Jarvis, skip this YouTube ad and toggle fullscreen.",
    "Jarvis, confirm yes execute to complete system verification.",
]


def get_input_device_name(device_index: int | None) -> str:
    import pyaudio

    pa = pyaudio.PyAudio()
    try:
        info = (
            pa.get_device_info_by_index(device_index)
            if device_index is not None
            else pa.get_default_input_device_info()
        )
        return str(info.get("name", "Default Microphone"))
    except Exception:
        return "Default Microphone"
    finally:
        pa.terminate()


def record_clip(
    duration_sec: float,
    sample_rate: int,
    chunk_samples: int,
    device_index: int | None = None,
) -> np.ndarray:
    """Record fixed-duration mono int16 PCM audio from microphone."""
    import pyaudio

    pa = pyaudio.PyAudio()
    frames: list[bytes] = []
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=chunk_samples,
    )
    try:
        # Discard 250ms of initial stream warmup (prevents Bluetooth HFP start-clipping)
        warmup_chunks = max(1, int(np.ceil((0.25 * sample_rate) / chunk_samples)))
        for _ in range(warmup_chunks):
            stream.read(chunk_samples, exception_on_overflow=False)

        total_chunks = max(1, int(np.ceil((duration_sec * sample_rate) / chunk_samples)))
        for _ in range(total_chunks):
            data = stream.read(chunk_samples, exception_on_overflow=False)
            frames.append(data)
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

    raw = b"".join(frames)
    return np.frombuffer(raw, dtype=np.int16).copy()


def run_enrollment(
    num_phrases: int,
    duration_sec: float,
    engine: str | None,
    output_path: str | None,
    verify_after: bool,
    device_index: int | None = None,
) -> int:
    cfg = load_config()
    if engine:
        cfg.biometrics.engine = engine
    if output_path:
        cfg.biometrics.voiceprint_path = output_path
    if device_index is not None:
        cfg.audio.input_device_index = device_index

    if not (5 <= num_phrases <= 10):
        print("Error: --phrases must be between 5 and 10 per PRD FR-1.", file=sys.stderr)
        return 2

    gate = VoiceBiometricGate(config=cfg)
    mic_name = get_input_device_name(cfg.audio.input_device_index)
    print(f"\n=== Project Jarvis — Voice Biometric Enrollment ({cfg.biometrics.engine}) ===")
    print(f"Input microphone   : {mic_name}")
    print(f"Target voiceprint  : {gate.voiceprint_path}")
    print(f"Phrases to record  : {num_phrases} ({duration_sec:.1f}s each)\n")
    print("Loading speaker embedding model (first run may download weights)...")
    _ = gate.backend
    print("Model ready.\n")

    clips: list[np.ndarray] = []
    for idx in range(num_phrases):
        prompt = CALIBRATION_PHRASES[idx % len(CALIBRATION_PHRASES)]
        while True:
            input(f"[{idx + 1}/{num_phrases}] Press ENTER, then clearly speak:\n  \"{prompt}\"\n> ")
            print("  Recording...", end="", flush=True)
            clip = record_clip(
                duration_sec=duration_sec,
                sample_rate=cfg.audio.sample_rate,
                chunk_samples=cfg.audio.chunk_samples,
                device_index=cfg.audio.input_device_index,
            )
            f32 = pcm16_to_float32(clip)
            rms = compute_rms_dbfs(f32)
            print(f" done (RMS: {rms:.1f} dBFS).")
            if rms < -50.0:
                print("  Too quiet or no speech detected. Please speak closer to the mic and retry.\n")
                continue
            clips.append(clip)
            break

    print("\nComputing master speaker embedding and validating consistency...")
    centroid, sims, mean_sim = gate.enroll_from_clips(clips, sample_rate=cfg.audio.sample_rate)
    for idx, sim in enumerate(sims, start=1):
        flag = "OK" if sim >= cfg.biometrics.min_phrase_consistency else "LOW"
        print(f"  Phrase {idx:02d} similarity to centroid: {sim:.4f} [{flag}]")

    saved_path = gate.save_voiceprint(
        centroid,
        phrase_count=len(clips),
        mean_consistency=mean_sim,
    )
    print(f"\nEnrolled voiceprint saved to: {saved_path}")
    print(f"Mean intra-speaker consistency: {mean_sim:.4f}")
    print(f"Configured verification threshold: {gate.threshold:.2f}\n")

    if verify_after:
        print("=== Live Verification Check ===")
        print("Speak any command (yours should PASS >= 0.78; another person should FAIL < 0.78).")
        input("Press ENTER and speak for 3.5 seconds...\n> ")
        test_clip = record_clip(
            duration_sec=3.5,
            sample_rate=cfg.audio.sample_rate,
            chunk_samples=cfg.audio.chunk_samples,
            device_index=cfg.audio.input_device_index,
        )
        res = gate.verify(test_clip, sample_rate=cfg.audio.sample_rate)
        status = "ACCEPTED (Owner)" if res.verified else "REJECTED (Non-Owner)"
        print(
            f"Result: {status} | similarity={res.similarity:.4f} | "
            f"threshold={res.threshold:.2f} | latency={res.latency_ms:.1f} ms"
        )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enroll owner voiceprint (owner_voiceprint.npy) for Project Jarvis."
    )
    parser.add_argument(
        "--phrases",
        type=int,
        default=7,
        help="Number of calibration phrases to record (5-10, default: 7).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=4.0,
        help="Recording window per phrase in seconds (default: 4.0).",
    )
    parser.add_argument(
        "--engine",
        choices=["speechbrain", "resemblyzer"],
        default=None,
        help="Biometric model backend (default: from config/settings.yaml).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .npy path (default: owner_voiceprint.npy).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run a live verification test immediately after enrollment.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Optional PyAudio input device index.",
    )
    args = parser.parse_args()
    return run_enrollment(
        num_phrases=args.phrases,
        duration_sec=args.duration,
        engine=args.engine,
        output_path=args.output,
        verify_after=args.verify,
        device_index=args.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
