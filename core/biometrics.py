"""Speaker biometric verification engine for Project Jarvis.

Supports SpeechBrain (ECAPA-TDNN) and Resemblyzer (GE2E) vocal embeddings
with strict L2-normalized cosine similarity gating (>= 0.78 default).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from config.settings import BiometricsConfig, JarvisConfig, load_config

logger = logging.getLogger(__name__)

SPEAKER_THRESHOLD = 0.60


@dataclass(frozen=True)
class BiometricResult:
    """Result of a speaker verification check."""

    verified: bool
    similarity: float
    threshold: float
    engine: str
    reason: str
    latency_ms: float


class EmbeddingBackend(Protocol):
    """Protocol for pluggable speaker embedding extractors."""

    name: str
    embedding_dim: int

    def extract_embedding(self, audio_f32: np.ndarray, sample_rate: int = 16000) -> np.ndarray: ...


def pcm16_to_float32(audio: np.ndarray) -> np.ndarray:
    """Convert int16 PCM or numeric array to float32 in [-1.0, 1.0]."""
    arr = np.asarray(audio)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float32)
    if np.issubdtype(arr.dtype, np.integer):
        return (arr.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
    return arr.astype(np.float32).clip(-1.0, 1.0)


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Return L2-normalized float32 vector (or zeros if norm is ~0)."""
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(v))
    if norm < 1e-8:
        return np.zeros_like(v, dtype=np.float32)
    return (v / norm).astype(np.float32)


def cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Compute cosine similarity in [-1.0, 1.0] between two embedding vectors."""
    a = l2_normalize(emb_a)
    b = l2_normalize(emb_b)
    if a.shape != b.shape or a.size == 0:
        return 0.0
    if float(np.linalg.norm(a)) < 1e-6 or float(np.linalg.norm(b)) < 1e-6:
        return 0.0
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def compute_rms_dbfs(audio_f32: np.ndarray) -> float:
    """Compute RMS level in dBFS."""
    if audio_f32.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(audio_f32), dtype=np.float64)))
    if rms < 1e-9:
        return -120.0
    return float(20.0 * np.log10(rms))


def resample_to_16k_f32(audio_f32: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Resample incoming mono float32 audio cleanly to 16 kHz PCM float32."""
    arr = np.asarray(audio_f32, dtype=np.float32).reshape(-1)
    if arr.size == 0 or sample_rate == 16000 or sample_rate <= 0:
        return arr
    from math import gcd
    from scipy.signal import resample_poly

    g = gcd(16000, int(sample_rate))
    up = 16000 // g
    down = int(sample_rate) // g
    resampled = resample_poly(arr, up, down).astype(np.float32)
    return np.clip(resampled, -1.0, 1.0)


def normalize_audio_f32(
    audio_f32: np.ndarray,
    target_rms_dbfs: float = -20.0,
    peak_limit: float = 0.90,
) -> np.ndarray:
    """Remove DC offset and apply RMS + peak normalization for consistent embeddings."""
    if audio_f32.size == 0:
        return audio_f32
    centered = audio_f32.astype(np.float32) - float(np.mean(audio_f32))
    cur_rms_db = compute_rms_dbfs(centered)
    if cur_rms_db <= -80.0:
        return centered
    gain = float(10.0 ** ((target_rms_dbfs - cur_rms_db) / 20.0))
    scaled = centered * min(gain, 25.0)
    peak = float(np.max(np.abs(scaled)))
    if peak > peak_limit and peak > 1e-8:
        scaled = scaled * (peak_limit / peak)
    return scaled.astype(np.float32)


def trim_silence_vad(audio_f32: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Trim leading/trailing and long internal silence using WebRTC VAD + energy gate.

    ECAPA-TDNN attentive statistics pooling degrades when silence dominates a clip;
    keeping only active speech frames yields higher and more consistent owner scores.
    """
    if audio_f32.size < int(sample_rate * 0.25):
        return audio_f32

    pcm16 = (np.clip(audio_f32, -1.0, 1.0) * 32767.0).astype(np.int16)
    frame_len = int(sample_rate * 0.02)  # 20 ms
    if frame_len <= 0 or pcm16.size < frame_len:
        return audio_f32

    vad = None
    try:
        import webrtcvad

        vad = webrtcvad.Vad(2)
    except Exception:
        pass

    voiced_flags: list[bool] = []
    for start in range(0, pcm16.size - frame_len + 1, frame_len):
        sub = pcm16[start : start + frame_len]
        sub_f32 = audio_f32[start : start + frame_len]
        if compute_rms_dbfs(sub_f32) < -48.0:
            voiced_flags.append(False)
        elif vad is not None:
            voiced_flags.append(bool(vad.is_speech(sub.tobytes(), sample_rate)))
        else:
            voiced_flags.append(True)

    if not any(voiced_flags):
        return audio_f32

    # Expand voiced mask by 4 frames (80 ms) on each side to preserve word edges
    collar = 4
    keep = [False] * len(voiced_flags)
    for i, flag in enumerate(voiced_flags):
        if flag:
            for j in range(max(0, i - collar), min(len(voiced_flags), i + collar + 1)):
                keep[j] = True

    chunks = [
        audio_f32[i * frame_len : (i + 1) * frame_len]
        for i, k in enumerate(keep)
        if k
    ]
    if not chunks:
        return audio_f32
    trimmed = np.concatenate(chunks)
    if trimmed.size < int(sample_rate * 0.25):
        return audio_f32
    return trimmed


class ResemblyzerBackend:
    """Resemblyzer GE2E 256-dim speaker embedding extractor."""

    name = "resemblyzer"
    embedding_dim = 256

    def __init__(self) -> None:
        from resemblyzer import VoiceEncoder, preprocess_wav

        self._preprocess_wav = preprocess_wav
        self._encoder = VoiceEncoder(device="cpu")
        logger.info("Loaded Resemblyzer VoiceEncoder on CPU.")

    def extract_embedding(self, audio_f32: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        wav = self._preprocess_wav(audio_f32, source_sr=sample_rate)
        if wav.size < int(sample_rate * 0.25):
            # Pad short utterances slightly after preprocessing so mel-spec succeeds
            wav = np.pad(wav, (0, max(0, int(sample_rate * 0.35) - wav.size)))
        emb = self._encoder.embed_utterance(wav)
        return l2_normalize(emb)


class SpeechBrainBackend:
    """SpeechBrain ECAPA-TDNN 192-dim speaker embedding extractor."""

    name = "speechbrain"
    embedding_dim = 192

    def __init__(self, source: str, savedir: Path) -> None:
        import torch

        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:  # pragma: no cover - legacy speechbrain compatibility
            from speechbrain.pretrained import EncoderClassifier

        savedir.mkdir(parents=True, exist_ok=True)
        self._torch = torch
        self._classifier = EncoderClassifier.from_hparams(
            source=source,
            savedir=str(savedir),
            run_opts={"device": "cpu"},
        )
        logger.info("Loaded SpeechBrain ECAPA-TDNN (%s) on CPU.", source)

    def extract_embedding(self, audio_f32: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        wav_16k = resample_to_16k_f32(audio_f32, sample_rate=sample_rate)
        wav = trim_silence_vad(wav_16k, sample_rate=16000)
        wav = normalize_audio_f32(wav)
        if wav.size < int(16000 * 0.35):
            wav = np.pad(wav, (0, int(16000 * 0.35) - wav.size))
        tensor = self._torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
        with self._torch.no_grad():
            emb = self._classifier.encode_batch(tensor)
        vec = emb.squeeze().cpu().numpy()
        return l2_normalize(vec)


def create_biometric_backend(cfg: JarvisConfig) -> EmbeddingBackend:
    """Instantiate configured biometric embedding backend with clean fallback."""
    engine = cfg.biometrics.engine.lower()
    if engine == "speechbrain":
        try:
            savedir = cfg.resolve_path(cfg.biometrics.speechbrain_savedir)
            return SpeechBrainBackend(cfg.biometrics.speechbrain_source, savedir)
        except Exception as exc:
            logger.warning(
                "SpeechBrain unavailable (%s); falling back to Resemblyzer.", exc
            )
            return ResemblyzerBackend()
    if engine == "resemblyzer":
        return ResemblyzerBackend()
    raise ValueError(f"Unsupported biometric engine: {cfg.biometrics.engine!r}")


class VoiceBiometricGate:
    """Real-time speaker verification gate backed by enrolled owner_voiceprint.npy."""

    def __init__(
        self,
        config: Optional[JarvisConfig] = None,
        backend: Optional[EmbeddingBackend] = None,
        voiceprint: Optional[np.ndarray] = None,
    ) -> None:
        self.config = config or load_config()
        self.bio_cfg: BiometricsConfig = self.config.biometrics
        self.voiceprint_path: Path = self.config.resolve_path(self.bio_cfg.voiceprint_path)
        self.meta_path: Path = self.voiceprint_path.with_suffix(".meta.json")
        self._backend: Optional[EmbeddingBackend] = backend
        self._owner_embedding: Optional[np.ndarray] = (
            l2_normalize(voiceprint) if voiceprint is not None else None
        )
        self._metadata: dict = {}

        if self._owner_embedding is None and self.voiceprint_path.exists():
            self.load_voiceprint()

    @property
    def backend(self) -> EmbeddingBackend:
        if self._backend is None:
            self._backend = create_biometric_backend(self.config)
        return self._backend

    @property
    def is_enrolled(self) -> bool:
        return self._owner_embedding is not None and self._owner_embedding.size > 0

    @property
    def threshold(self) -> float:
        return float(self.bio_cfg.similarity_threshold)

    def load_voiceprint(self) -> bool:
        """Load enrolled owner voiceprint from disk (returns True if valid)."""
        if not self.voiceprint_path.exists():
            logger.warning("Owner voiceprint not found at %s", self.voiceprint_path)
            self._owner_embedding = None
            return False

        raw = np.load(self.voiceprint_path)
        normed = l2_normalize(raw)
        if normed.size == 0 or float(np.linalg.norm(normed)) < 1e-6:
            logger.error("Voiceprint at %s is empty or zero-norm.", self.voiceprint_path)
            self._owner_embedding = None
            return False

        self._owner_embedding = normed
        if self.meta_path.exists():
            try:
                self._metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))
                meta_engine = self._metadata.get("engine")
                if meta_engine and meta_engine != self.bio_cfg.engine and self._backend is None:
                    logger.info(
                        "Aligning biometric engine to enrolled voiceprint engine: %s",
                        meta_engine,
                    )
                    self.bio_cfg.engine = meta_engine
            except Exception as exc:
                logger.debug("Could not parse voiceprint metadata: %s", exc)
        logger.info(
            "Loaded owner voiceprint (%d dims) from %s",
            self._owner_embedding.size,
            self.voiceprint_path,
        )
        return True

    def save_voiceprint(
        self,
        embedding: np.ndarray,
        phrase_count: int,
        mean_consistency: float,
    ) -> Path:
        """Persist normalized owner voiceprint vector and companion metadata."""
        normed = l2_normalize(embedding)
        self.voiceprint_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.voiceprint_path, normed)

        meta = {
            "engine": self.backend.name,
            "embedding_dim": int(normed.size),
            "phrase_count": int(phrase_count),
            "mean_consistency": round(float(mean_consistency), 4),
            "threshold": float(self.threshold),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self._owner_embedding = normed
        self._metadata = meta
        return self.voiceprint_path

    def enroll_from_clips(
        self,
        clips: list[np.ndarray],
        sample_rate: int = 16000,
    ) -> tuple[np.ndarray, list[float], float]:
        """Compute master owner embedding from 5-10 calibration audio clips.

        Returns:
            (master_embedding, per_clip_similarities, mean_consistency)
        """
        if len(clips) < 3:
            raise ValueError("At least 3 speech clips are required for enrollment.")

        embeddings: list[np.ndarray] = []
        for idx, clip in enumerate(clips):
            f32 = pcm16_to_float32(clip)
            if compute_rms_dbfs(f32) < -55.0:
                raise ValueError(f"Calibration clip #{idx + 1} is too quiet / silent.")
            emb = self.backend.extract_embedding(f32, sample_rate=sample_rate)
            embeddings.append(emb)

        centroid = l2_normalize(np.mean(np.stack(embeddings, axis=0), axis=0))
        sims = [cosine_similarity(e, centroid) for e in embeddings]

        # Reject gross outliers if enough clips remain (>= 5)
        min_cons = float(self.bio_cfg.min_phrase_consistency)
        kept = [e for e, s in zip(embeddings, sims) if s >= min_cons]
        if len(kept) >= max(3, len(embeddings) // 2):
            centroid = l2_normalize(np.mean(np.stack(kept, axis=0), axis=0))
            sims = [cosine_similarity(e, centroid) for e in embeddings]

        mean_consistency = float(np.mean(sims))
        return centroid, sims, mean_consistency

    def verify(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> BiometricResult:
        """Verify incoming audio against the enrolled owner voiceprint.

        Rejects silence, uninitialized voiceprints, dimension mismatches,
        and any speaker scoring below `similarity_threshold` (default 0.78).
        """
        t0 = time.perf_counter()
        if not self.is_enrolled or self._owner_embedding is None:
            return BiometricResult(
                verified=False,
                similarity=0.0,
                threshold=self.threshold,
                engine=self.bio_cfg.engine,
                reason="no_enrolled_voiceprint",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        f32 = resample_to_16k_f32(pcm16_to_float32(audio), sample_rate=sample_rate)
        min_samples = int(16000 * max(0.2, self.config.audio.min_speech_seconds))
        if f32.size < min_samples:
            return BiometricResult(
                verified=False,
                similarity=0.0,
                threshold=self.threshold,
                engine=self.bio_cfg.engine,
                reason="audio_too_short",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        rms_db = compute_rms_dbfs(f32)
        if rms_db < -55.0:
            return BiometricResult(
                verified=False,
                similarity=0.0,
                threshold=self.threshold,
                engine=self.bio_cfg.engine,
                reason="silence_or_low_energy",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        embedding = self.backend.extract_embedding(f32, sample_rate=sample_rate)
        if embedding.shape != self._owner_embedding.shape:
            return BiometricResult(
                verified=False,
                similarity=0.0,
                threshold=self.threshold,
                engine=self.backend.name,
                reason=(
                    f"dimension_mismatch:{embedding.size}_vs_{self._owner_embedding.size}"
                ),
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        sim = cosine_similarity(embedding, self._owner_embedding)
        verified = sim >= self.threshold
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        reason = "verified" if verified else "below_threshold"
        return BiometricResult(
            verified=verified,
            similarity=sim,
            threshold=self.threshold,
            engine=self.backend.name,
            reason=reason,
            latency_ms=elapsed_ms,
        )
