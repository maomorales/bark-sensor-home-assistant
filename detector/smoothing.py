"""Event smoothing and cooldown logic."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque


@dataclass
class SmootherConfig:
    window_count: int
    positives_required: int
    cooldown_seconds: float


@dataclass
class EventSmoother:
    """Implements majority vote with cooldown."""

    config: SmootherConfig
    _history: Deque[bool] = field(default_factory=deque)
    _last_trigger_ts: float = 0.0

    def update(self, is_positive: bool, timestamp: float | None = None) -> bool:
        """
        Update the smoother with the latest decision.

        Returns True if the event should trigger, subject to cooldown.
        """
        ts = timestamp if timestamp is not None else time.time()
        cfg = self.config

        if len(self._history) == cfg.window_count:
            self._history.popleft()
        self._history.append(is_positive)

        positives = sum(1 for flag in self._history if flag)
        enough_votes = positives >= cfg.positives_required
        cooldown_active = (ts - self._last_trigger_ts) < cfg.cooldown_seconds

        if enough_votes and not cooldown_active:
            self._last_trigger_ts = ts
            self._history.clear()
            return True
        return False

    def reset(self) -> None:
        """Clear history and cooldown."""
        self._history.clear()
        self._last_trigger_ts = 0.0
