"""Audio capture ring buffer and WAV export."""

from __future__ import annotations

import threading
import time
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
    # Peak-normalise saved clips to this level so distant, quiet events are
    # audible on playback. Affects the saved file only, never detection, which
    # always scores the unmodified audio. 0 disables.
    normalize_peak: float = 0.0
    max_age_hours: float = 0
    # Hard ceiling on the capture directory. Age alone bounds nothing: a noisy
    # night can produce thousands of files that are all younger than the age
    # limit. 0 disables the size check.
    max_total_mb: float = 0


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
        self._cleanup_stop = threading.Event()
        self._cleanup_thread: Optional[threading.Thread] = None
        # Serialises deletion. The hourly sweep and the post-write check can
        # otherwise both scan, both compute a total from the same stale view,
        # and each delete down to the budget -- removing roughly twice what is
        # needed.
        self._prune_lock = threading.Lock()
        self._ensure_output_dir()
        self._start_cleanup_loop()

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

        applied_gain = 1.0
        if self.config.normalize_peak > 0:
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            if peak > 1e-6:
                applied_gain = self.config.normalize_peak / peak
                audio = np.clip(audio * applied_gain, -1.0, 1.0)

        int_audio = np.int16(audio * 32767)
        wavfile.write(job.file_path, self.sample_rate, int_audio)
        if applied_gain != 1.0:
            logger.info(
                "Saved capture to {} (playback gain {:.0f}x applied)",
                job.file_path,
                applied_gain,
            )
        else:
            logger.info("Saved capture to {}", job.file_path)
        # Enforce on every write, not just on the hourly sweep -- a burst of
        # events can blow the budget many times over between two sweeps.
        self._enforce_size_budget()

    # Cleanup -----------------------------------------------------------

    def _enforce_size_budget(self) -> None:
        """Delete the oldest captures until the directory fits the budget."""
        if self._disabled or self.config.max_total_mb <= 0:
            return
        with self._prune_lock:
            self._enforce_size_budget_locked()

    def _enforce_size_budget_locked(self) -> None:
        budget_bytes = self.config.max_total_mb * 1024 * 1024
        try:
            files = []
            total = 0
            for path in self.config.out_dir.glob("*.wav"):
                try:
                    stat = path.stat()
                except OSError:
                    continue
                files.append((stat.st_mtime, stat.st_size, path))
                total += stat.st_size
        except OSError as exc:
            logger.warning("Capture size scan failed: {}", exc)
            return

        if total <= budget_bytes:
            return

        files.sort()  # oldest first
        removed = 0
        freed = 0
        for _mtime, size, path in files:
            if total <= budget_bytes:
                break
            try:
                path.unlink()
                total -= size
                freed += size
                removed += 1
            except OSError as exc:
                logger.warning("Could not remove {}: {}", path, exc)

        if removed:
            logger.warning(
                "Capture directory exceeded {} MB; removed {} oldest file(s), freed {:.1f} MB",
                self.config.max_total_mb,
                removed,
                freed / (1024 * 1024),
            )

    def _start_cleanup_loop(self) -> None:
        if not self.config.enabled or self._disabled:
            return
        if self.config.max_age_hours <= 0 and self.config.max_total_mb <= 0:
            return
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True, name="capture-cleanup"
        )
        self._cleanup_thread.start()

    def _cleanup_loop(self) -> None:
        interval = 3600.0  # check once per hour
        while not self._cleanup_stop.wait(timeout=0):
            self._cleanup_old_captures()
            if self._cleanup_stop.wait(timeout=interval):
                break

    def _cleanup_old_captures(self) -> None:
        self._enforce_size_budget()
        if self.config.max_age_hours <= 0:
            return

        with self._prune_lock:
            self._cleanup_old_captures_locked()

    def _cleanup_old_captures_locked(self) -> None:
        max_age_secs = self.config.max_age_hours * 3600
        now = time.time()
        removed = 0
        try:
            for path in self.config.out_dir.glob("*.wav"):
                try:
                    if now - path.stat().st_mtime > max_age_secs:
                        path.unlink()
                        removed += 1
                        logger.debug("Removed old capture {}", path)
                except OSError as exc:
                    logger.warning("Could not remove {}: {}", path, exc)
        except OSError as exc:
            logger.warning("Cleanup scan failed: {}", exc)
        if removed:
            logger.info("Cleaned up {} old capture(s)", removed)

    def stop_cleanup(self) -> None:
        """Signal the cleanup thread to stop."""
        self._cleanup_stop.set()
