#!/usr/bin/env python3
"""Translate Home Assistant add-on options into the detector's config file.

The add-on UI exposes a flat, friendly set of options; the detector expects the
nested YAML documented in ``config/example-config.yaml``. Secrets are
deliberately not written here -- MQTT credentials reach the app through
environment variables set by the service ``run`` script.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

# Captures land in /media so they show up in Home Assistant's Media browser and
# can be played back in the UI. /share is not browsable there, and /data would
# be private to this add-on.
CAPTURE_DIR = "/media/barkdetector"


def resolve_mic_device(raw: object) -> object:
    """Accept either a PortAudio device index or a substring of its name.

    The add-on option is a string because the HA options UI has no "int or
    text" type; an all-digit value is treated as an index, anything else is
    passed through for name matching, and empty means "system default".
    """
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    if value.isdigit():
        return int(value)
    return value


def build_config(options: dict) -> dict:
    mode = str(options.get("detection_mode", "yamnet")).lower()
    threshold = float(options.get("conf_threshold", 0.2))

    return {
        "device_id": options.get("device_id", "barkdetector"),
        "audio": {
            "sample_rate": 16000,
            "channels": 1,
            "window_seconds": 0.975,
            "hop_seconds": 0.5,
            "mic_device_index": resolve_mic_device(options.get("mic_device")),
        },
        "detection": {
            "mode": mode,
            "yamnet": {
                "model_url": "https://storage.googleapis.com/audioset/yamnet/yamnet.tflite",
                "classes_url": (
                    "https://raw.githubusercontent.com/tensorflow/models/master/"
                    "research/audioset/yamnet/yamnet_class_map.csv"
                ),
                "conf_threshold": threshold,
                "label_substrings": ["dog", "bark", "yip", "bow-wow", "howl"],
            },
            "normalize": {
                "enabled": bool(options.get("normalize_windows", False)),
                "target_peak": 0.5,
                "noise_floor": float(options.get("normalize_noise_floor", 0.005)),
                "max_gain": float(options.get("normalize_max_gain", 30)),
            },
            "heuristic": {
                "rms_threshold": 0.015,
                "band_low_hz": 400,
                "band_high_hz": 3000,
                "band_energy_min": 5.0e-6,
            },
        },
        "smoothing": {
            "window_count": int(options.get("windows_total", 5)),
            "positives_required": int(options.get("windows_required", 3)),
            "cooldown_seconds": int(options.get("cooldown_seconds", 10)),
        },
        "capture": {
            "enabled": bool(options.get("capture_enabled", False)),
            "ring_seconds": 20,
            "pre_seconds": 5,
            "post_seconds": 5,
            "normalize_peak": float(options.get("capture_normalize_peak", 0.7)),
            "out_dir": CAPTURE_DIR,
            "max_age_hours": float(options.get("capture_retention_hours", 24)),
            "max_total_mb": float(options.get("capture_max_mb", 512)),
        },
        # Placeholders only. The real values come from BARKDETECTOR_MQTT_*.
        "mqtt": {
            "host": "",
            "port": 1883,
            "topic": options.get("mqtt_topic", "home/sensors/dog_bark"),
            "username": "",
            "password": "",
            "client_id": "",
        },
        "dailybot": {"workflow_url": ""},
        "logging": {
            "level": str(options.get("log_level", "info")).upper(),
            # s6 captures stdout/stderr into the add-on log; a second copy on
            # disk would grow unbounded inside the container.
            "file_path": "/data/barkdetector.log",
        },
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: render_config.py <options.json> <output.yaml>", file=sys.stderr)
        return 2

    options_path, output_path = Path(sys.argv[1]), Path(sys.argv[2])

    try:
        options = json.loads(options_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Unable to read add-on options from {options_path}: {exc}", file=sys.stderr)
        return 1

    config = build_config(options)

    if config["capture"]["enabled"]:
        Path(CAPTURE_DIR).mkdir(parents=True, exist_ok=True)

    output_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Rendered detector configuration to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
