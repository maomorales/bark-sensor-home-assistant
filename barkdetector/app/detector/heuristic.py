"""Simple heuristic bark detector based on signal energy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.signal import welch


@dataclass
class HeuristicConfig:
    rms_threshold: float
    band_low_hz: float
    band_high_hz: float
    band_energy_min: float


class HeuristicBarkDetector:
    """Fallback detector relying on RMS and band energy heuristics."""

    def __init__(self, config: HeuristicConfig, sample_rate: int) -> None:
        self.config = config
        self.sample_rate = sample_rate
        self.last_metrics: Dict[str, float] | None = None

    def score_bark(self, samples: np.ndarray) -> float:
        """Return a heuristic bark score in the range [0, 1]."""
        score, _ = self._compute(samples)
        return score

    def evaluate(self, samples: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Return both score and metrics."""
        return self._compute(samples)

    def is_positive(self, metrics: Dict[str, float]) -> bool:
        """Determine if heuristics satisfy threshold conditions."""
        return bool(
            metrics["rms"] >= self.config.rms_threshold
            and metrics["band_energy"] >= self.config.band_energy_min
        )

    def _compute(self, samples: np.ndarray) -> Tuple[float, Dict[str, float]]:
        samples = samples.astype(np.float32, copy=False)
        rms = float(np.sqrt(np.mean(np.square(samples))))

        freqs, psd = welch(samples, fs=self.sample_rate, nperseg=1024)
        band_mask = (freqs >= self.config.band_low_hz) & (freqs <= self.config.band_high_hz)
        if not np.any(band_mask):
            band_energy = 0.0
        else:
            band_energy = float(np.trapz(psd[band_mask], freqs[band_mask]))

        rms_ratio = np.clip(
            (rms - self.config.rms_threshold) / max(self.config.rms_threshold, 1e-8) + 0.5,
            0.0,
            1.0,
        )
        band_ratio = np.clip(
            band_energy / max(self.config.band_energy_min, 1e-12),
            0.0,
            1.5,
        )
        score = float(np.clip((rms_ratio * 0.6 + min(band_ratio, 1.0) * 0.4), 0.0, 1.0))

        metrics = {
            "rms": rms,
            "band_energy": band_energy,
            "rms_ratio": rms_ratio,
            "band_ratio": band_ratio,
        }
        self.last_metrics = metrics
        return score, metrics
