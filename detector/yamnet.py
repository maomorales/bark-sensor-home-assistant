"""YAMNet-based bark detector."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import requests
from loguru import logger
from tflite_runtime.interpreter import Interpreter


class YAMNetInitializationError(Exception):
    """Raised when the YAMNet detector cannot be initialised."""


@dataclass
class YAMNetConfig:
    model_url: str
    classes_url: str
    conf_threshold: float
    label_substrings: Iterable[str]


class YAMNetBarkDetector:
    """Wraps a TFLite YAMNet model for bark detection."""

    def __init__(self, config: YAMNetConfig, models_dir: Path | None = None) -> None:
        self.config = config
        self.models_dir = models_dir or Path(__file__).resolve().parents[1] / "models"
        self.model_path = self.models_dir / "yamnet.tflite"
        self.classes_path = self.models_dir / "yamnet_class_map.csv"
        self._interpreter: Interpreter | None = None
        self._bark_indices: List[int] = []
        self._input_index: int | None = None
        self._output_index: int | None = None
        self._current_input_length: int | None = None

        try:
            self._prepare_files()
            self._load_interpreter()
            self._load_class_map()
        except Exception as exc:  # pragma: no cover - defensive
            raise YAMNetInitializationError(str(exc)) from exc

    def score_bark(self, samples: np.ndarray) -> float:
        """Return the bark confidence score derived from YAMNet outputs."""
        if self._interpreter is None or self._input_index is None or self._output_index is None:
            raise RuntimeError("YAMNet interpreter is not initialised")

        waveform = samples.astype(np.float32, copy=False)
        if waveform.ndim != 1:
            waveform = waveform.squeeze()

        if self._current_input_length != waveform.shape[0]:
            self._resize_input(len(waveform))

        self._interpreter.set_tensor(self._input_index, waveform)
        self._interpreter.invoke()
        predictions = self._interpreter.get_tensor(self._output_index)

        if not self._bark_indices:
            return 0.0

        bark_probs = predictions[:, self._bark_indices]
        score = float(np.max(bark_probs)) if bark_probs.size else 0.0
        return float(np.clip(score, 0.0, 1.0))

    # Internal helpers -------------------------------------------------

    def _prepare_files(self) -> None:
        os.makedirs(self.models_dir, exist_ok=True)
        if not self.model_path.exists():
            logger.info("Downloading YAMNet model from {}", self.config.model_url)
            self._download_file(self.config.model_url, self.model_path)
        if not self.classes_path.exists():
            logger.info("Downloading YAMNet class map from {}", self.config.classes_url)
            self._download_file(self.config.classes_url, self.classes_path)

    def _download_file(self, url: str, destination: Path) -> None:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        tmp_path = destination.with_suffix(destination.suffix + ".tmp")
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)
        tmp_path.replace(destination)

    def _load_interpreter(self) -> None:
        self._interpreter = Interpreter(model_path=str(self.model_path))
        self._interpreter.allocate_tensors()
        input_details = self._interpreter.get_input_details()[0]
        output_details = self._interpreter.get_output_details()[0]
        self._input_index = int(input_details["index"])
        self._output_index = int(output_details["index"])
        self._current_input_length = input_details["shape"][0] or None

    def _resize_input(self, length: int) -> None:
        if self._interpreter is None or self._input_index is None:
            raise RuntimeError("Interpreter not prepared")
        self._interpreter.resize_tensor_input(self._input_index, [length], strict=False)
        self._interpreter.allocate_tensors()
        self._current_input_length = length

    def _load_class_map(self) -> None:
        substrings = [s.lower() for s in self.config.label_substrings]
        with self.classes_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            labels = [row.get("display_name", "").strip() for row in reader]

        if not labels:
            raise RuntimeError("YAMNet class map is empty")

        bark_indices: List[int] = []
        for idx, label in enumerate(labels):
            label_lower = label.lower()
            if any(sub in label_lower for sub in substrings):
                bark_indices.append(idx)

        if not bark_indices:
            logger.warning(
                "No YAMNet classes matched substrings {}; bark detection will always be zero",
                substrings,
            )
        self._bark_indices = bark_indices
