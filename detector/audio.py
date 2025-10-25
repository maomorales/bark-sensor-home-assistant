"""Audio capture utilities for bark detection."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple

import numpy as np
import sounddevice as sd
from loguru import logger
from scipy.signal import resample_poly


@dataclass
class AudioStreamConfig:
    """Configuration for the audio stream."""

    sample_rate: int
    channels: int
    window_seconds: float
    hop_seconds: float
    mic_device_index: Optional[int] = None


class AudioStreamError(Exception):
    """Raised when the audio stream cannot be established."""


class AudioStreamProvider:
    """Provides resampled audio chunks ready for detection."""

    def __init__(self, config: AudioStreamConfig) -> None:
        self.config = config
        self._stop_event = threading.Event()
        self._source_rate: Optional[int] = None

    @staticmethod
    def list_input_devices() -> List[Tuple[int, str, float, int]]:
        """Return a list of available input devices."""
        devices = sd.query_devices()
        result: List[Tuple[int, str, float, int]] = []
        for idx, info in enumerate(devices):
            if info.get("max_input_channels", 0) > 0:
                result.append(
                    (
                        idx,
                        info.get("name", f"Device {idx}"),
                        float(info.get("default_samplerate", 0) or 0),
                        int(info.get("max_input_channels", 0)),
                    )
                )
        return result

    def stop(self) -> None:
        """Signal the audio stream to stop."""
        self._stop_event.set()

    def stream_chunks(self) -> Generator[np.ndarray, None, None]:
        """
        Yield audio chunks resampled to target sample rate.

        Each yielded chunk spans hop_seconds worth of audio at the target
        sample rate (e.g., 0.5 seconds -> 8000 samples at 16 kHz).
        """
        target_rate = self.config.sample_rate
        hop_samples_target = max(1, int(round(self.config.hop_seconds * target_rate)))

        while not self._stop_event.is_set():
            queue_: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=8)

            try:
                stream, source_rate = self._open_stream(queue_)
            except AudioStreamError as exc:
                logger.error("Audio stream setup failed: {}", exc)
                time.sleep(5.0)
                continue

            self._source_rate = source_rate

            logger.info(
                "Audio stream started at {} Hz (target {} Hz) using device {}",
                source_rate,
                target_rate,
                self.config.mic_device_index
                if self.config.mic_device_index is not None
                else "default",
            )

            try:
                with stream:
                    while not self._stop_event.is_set():
                        try:
                            chunk = queue_.get(timeout=1.0)
                        except queue.Empty:
                            continue

                        if chunk.size == 0:
                            continue

                        mono = self._to_mono(chunk)

                        if source_rate != target_rate:
                            resampled = resample_poly(mono, target_rate, source_rate)
                        else:
                            resampled = mono

                        if resampled.size == 0:
                            continue

                        if resampled.size != hop_samples_target:
                            # Pad or trim to maintain consistent chunk size.
                            resampled = self._pad_or_trim(resampled, hop_samples_target)

                        yield resampled.astype(np.float32, copy=False)
            except sd.PortAudioError as exc:
                logger.error("Audio stream encountered an error: {}", exc)
                time.sleep(2.0)
            finally:
                logger.info("Audio stream stopped; reconnecting in 2 seconds")
                time.sleep(2.0)

    def _open_stream(
        self, queue_: "queue.Queue[np.ndarray]"
    ) -> Tuple[sd.InputStream, int]:
        device = self.config.mic_device_index
        target_rate = self.config.sample_rate
        candidate_rates: List[int] = [target_rate]

        try:
            device_info = sd.query_devices(device, "input")
            default_rate = int(device_info.get("default_samplerate", 0) or 0)
            if default_rate and default_rate not in candidate_rates:
                candidate_rates.append(default_rate)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Unable to query device info for {}: {}", device, exc)

        for rate in candidate_rates:
            try:
                hop_samples = max(1, int(round(self.config.hop_seconds * rate)))
                stream = sd.InputStream(
                    device=device,
                    samplerate=rate,
                    blocksize=hop_samples,
                    channels=self.config.channels,
                    dtype="float32",
                    callback=self._make_callback(queue_),
                )
                return stream, rate
            except sd.PortAudioError as exc:
                logger.warning(
                    "Failed to open audio stream at {} Hz (device {}): {}",
                    rate,
                    device,
                    exc,
                )

        raise AudioStreamError("Unable to open microphone stream at any supported rate")

    @staticmethod
    def _make_callback(
        queue_: "queue.Queue[np.ndarray]",
    ) -> "sd.CallbackType":
        def callback(indata, frames, time_info, status):  # type: ignore[override]
            if status:
                logger.warning("Audio callback status: {}", status)
            try:
                queue_.put_nowait(indata.copy())
            except queue.Full:
                # Drop oldest sample to avoid blocking callback.
                try:
                    queue_.get_nowait()
                    queue_.put_nowait(indata.copy())
                except queue.Empty:
                    pass

        return callback

    def _to_mono(self, chunk: np.ndarray) -> np.ndarray:
        if chunk.ndim == 1 or chunk.shape[1] == 1:
            return np.squeeze(chunk).astype(np.float32, copy=False)
        return np.mean(chunk, axis=1).astype(np.float32, copy=False)

    @staticmethod
    def _pad_or_trim(samples: np.ndarray, target_size: int) -> np.ndarray:
        if samples.size == target_size:
            return samples
        if samples.size > target_size:
            return samples[:target_size]
        pad_width = target_size - samples.size
        return np.pad(samples, (0, pad_width), mode="constant")
