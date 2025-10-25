"""Audio capture ring buffer and WAV export."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, List, Optional

import numpy as np
from loguru import logger
from scipy.io import wavfile


@dataclass
class CaptureConfig:
    enabled: bool
    ring_seconds: float
    pre_seconds: float
    post_seconds: float
    out_dir: Path


@dataclass
class _CaptureJob:
    pre_audio: np.ndarray
    post_samples: int
    file_path: Path
    start_ts: float
    collected: List[np.ndarray] = field(default_factory=list)

    @property
    def collected_samples(self) -> int:
        return sum(chunk.size for chunk in self.collected)

    def add_samples(self, samples: np.ndarray) -> None:
        if self.collected_samples >= self.post_samples:
            return
        needed = self.post_samples - self.collected_samples
        if needed <= 0:
            return
        self.collected.append(samples[:needed].copy())

    def ready(self) -> bool:
        return self.collected_samples >= self.post_samples

    def final_audio(self) -> np.ndarray:
        post = np.concatenate(self.collected, axis=0) if self.collected else np.array([], dtype=np.float32)
        post = post[: self.post_samples]
        return np.concatenate([self.pre_audio, post], axis=0)


class AudioRingBuffer:
    """Fixed-size ring buffer for recent audio samples."""

    def __init__(self, capacity_samples: int) -> None:
        from collections import deque

        self.capacity = capacity_samples
        self._buffer: Deque[np.ndarray] = deque()
        self._total = 0

    def extend(self, samples: np.ndarray) -> None:
        chunk = samples.astype(np.float32, copy=True)
        self._buffer.append(chunk)
        self._total += chunk.size
        self._trim()

    def _trim(self) -> None:
        while self._total > self.capacity and self._buffer:
            excess = self._total - self.capacity
            left = self._buffer[0]
            if left.size <= excess:
                self._buffer.popleft()
                self._total -= left.size
            else:
                self._buffer[0] = left[excess:]
                self._total -= excess

    def recent(self, samples: int) -> np.ndarray:
        samples = min(samples, self._total)
        if samples <= 0:
            return np.zeros(0, dtype=np.float32)

        result = np.zeros(samples, dtype=np.float32)
        remaining = samples
        idx = samples
        for chunk in reversed(self._buffer):
            if remaining <= 0:
                break
            take = min(chunk.size, remaining)
            idx -= take
            result[idx : idx + take] = chunk[-take:]
            remaining -= take
        return result


class AudioCaptureManager:
    """Handles capture buffers and delayed WAV exports."""

    def __init__(
        self,
        config: CaptureConfig,
        sample_rate: int,
    ) -> None:
        self.config = config
        self.sample_rate = sample_rate
        self._ring = AudioRingBuffer(int(config.ring_seconds * sample_rate))
        self._jobs: List[_CaptureJob] = []
        self._disabled = False
        self._ensure_output_dir()

    def _ensure_output_dir(self) -> None:
        if not self.config.enabled:
            return
        try:
            self.config.out_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError as exc:
            logger.error("Capture directory {} is not writable: {}", self.config.out_dir, exc)
            self._disabled = True

    def extend(self, samples: np.ndarray) -> List[Path]:
        """Feed new samples into the ring buffer and active jobs."""
        self._ring.extend(samples)
        completed: List[Path] = []

        if self._disabled or not self.config.enabled:
            return completed

        for job in list(self._jobs):
            job.add_samples(samples)
            if job.ready():
                try:
                    self._write_job(job)
                    completed.append(job.file_path)
                except Exception as exc:  # pragma: no cover - file system dependent
                    logger.error("Failed to write capture {}: {}", job.file_path, exc)
                finally:
                    self._jobs.remove(job)
        return completed

    def schedule_capture(self, event_ts: float, device_id: str) -> Optional[Path]:
        """Schedule a capture around an event."""
        if self._disabled or not self.config.enabled:
            return None

        pre_samples = int(self.config.pre_seconds * self.sample_rate)
        post_samples = int(self.config.post_seconds * self.sample_rate)
        total_samples = pre_samples + post_samples
        if total_samples <= 0:
            return None

        timestamp = datetime.fromtimestamp(event_ts)
        filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{device_id}.wav"
        file_path = self.config.out_dir / filename

        pre_audio = self._ring.recent(pre_samples)
        job = _CaptureJob(pre_audio=pre_audio, post_samples=post_samples, file_path=file_path, start_ts=event_ts)
        if post_samples == 0:
            try:
                self._write_job(job)
                return file_path
            except Exception as exc:  # pragma: no cover - filesystem dependent
                logger.error("Failed to write immediate capture {}: {}", file_path, exc)
                return None

        self._jobs.append(job)
        return file_path

    def _write_job(self, job: _CaptureJob) -> None:
        audio = job.final_audio()
        audio = np.clip(audio, -1.0, 1.0)
        int_audio = np.int16(audio * 32767)
        wavfile.write(job.file_path, self.sample_rate, int_audio)
        logger.info("Saved capture to {}", job.file_path)
