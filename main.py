"""Entry point for the bark detector PoC."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml
from loguru import logger

from detector.audio import AudioStreamConfig, AudioStreamProvider
from detector.capture import AudioCaptureManager, CaptureConfig
from detector.heuristic import HeuristicBarkDetector, HeuristicConfig
from detector.smoothing import EventSmoother, SmootherConfig
from detector.yamnet import (
    YAMNetBarkDetector,
    YAMNetConfig,
    YAMNetInitializationError,
)
from mqtt.mqtt_client import MQTTConfig, MQTTPublisher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dog bark detector")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to configuration file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without publishing MQTT events",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration file must define a mapping")
    return config


def setup_logging(log_config: Dict[str, Any]) -> None:
    level = log_config.get("level", "INFO")
    logger.remove()
    logger.add(sys.stderr, level=level)

    file_path = Path(log_config.get("file_path", "barkdetector.log"))
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(file_path),
            level=level,
            rotation="5 MB",
            retention=5,
            enqueue=True,
            backtrace=False,
            diagnose=False,
        )
    except PermissionError as exc:
        logger.warning(
            "Unable to write log file at {} ({}). Continuing with console logging only.",
            file_path,
            exc,
        )


def list_devices() -> None:
    devices = AudioStreamProvider.list_input_devices()
    if not devices:
        print("No audio input devices detected.")
        return
    print("Available audio input devices:")
    for idx, name, default_rate, channels in devices:
        print(
            f"[{idx}] {name} - default_samplerate={default_rate:.0f}Hz, max_input_channels={channels}"
        )


def build_detectors(config: Dict[str, Any], sample_rate: int) -> tuple[Any, str, HeuristicBarkDetector, float]:
    detection_cfg = config.get("detection", {})
    mode = detection_cfg.get("mode", "yamnet").lower()
    heur_cfg = detection_cfg.get("heuristic", {})
    heuristic = HeuristicBarkDetector(
        HeuristicConfig(
            rms_threshold=float(heur_cfg.get("rms_threshold", 0.02)),
            band_low_hz=float(heur_cfg.get("band_low_hz", 400)),
            band_high_hz=float(heur_cfg.get("band_high_hz", 3000)),
            band_energy_min=float(heur_cfg.get("band_energy_min", 1.0e-6)),
        ),
        sample_rate=sample_rate,
    )

    yamnet_threshold = float(detection_cfg.get("yamnet", {}).get("conf_threshold", 0.6))
    active_detector: Any = heuristic
    active_name = "heuristic"

    if mode == "yamnet":
        yamnet_cfg = detection_cfg.get("yamnet", {})
        try:
            active_detector = YAMNetBarkDetector(
                YAMNetConfig(
                    model_url=yamnet_cfg.get(
                        "model_url",
                        "https://storage.googleapis.com/audioset/yamnet/yamnet.tflite",
                    ),
                    classes_url=yamnet_cfg.get(
                        "classes_url",
                        "https://storage.googleapis.com/audioset/yamnet/yamnet_class_map.csv",
                    ),
                    conf_threshold=yamnet_threshold,
                    label_substrings=yamnet_cfg.get(
                        "label_substrings",
                        ("dog", "bark", "bow", "yip"),
                    ),
                )
            )
            active_name = "yamnet"
            logger.info("YAMNet detector initialised successfully")
        except YAMNetInitializationError as exc:
            logger.warning("Failed to initialise YAMNet: {}. Falling back to heuristic.", exc)
            active_detector = heuristic
            active_name = "heuristic"
    elif mode != "heuristic":
        logger.warning("Unknown detector mode '{}', defaulting to heuristic", mode)

    return active_detector, active_name, heuristic, yamnet_threshold


def configure_capture(config: Dict[str, Any], sample_rate: int) -> AudioCaptureManager:
    capture_cfg = config.get("capture", {})
    enabled = bool(capture_cfg.get("enabled", True))
    out_dir = Path(capture_cfg.get("out_dir", "/var/lib/barkdetector/captures"))
    capture_manager = AudioCaptureManager(
        CaptureConfig(
            enabled=enabled,
            ring_seconds=float(capture_cfg.get("ring_seconds", 20)),
            pre_seconds=float(capture_cfg.get("pre_seconds", 5)),
            post_seconds=float(capture_cfg.get("post_seconds", 5)),
            out_dir=out_dir,
        ),
        sample_rate=sample_rate,
    )
    return capture_manager


def build_mqtt(config: Dict[str, Any], dry_run: bool) -> Optional[MQTTPublisher]:
    mqtt_cfg = config.get("mqtt", {})
    if dry_run:
        logger.info("Dry run enabled; MQTT publishing is disabled")
        return None

    publisher = MQTTPublisher(
        MQTTConfig(
            host=mqtt_cfg.get("host", "localhost"),
            port=int(mqtt_cfg.get("port", 1883)),
            topic=mqtt_cfg.get("topic", "home/sensors/dog_bark"),
            username=(mqtt_cfg.get("username") or None),
            password=(mqtt_cfg.get("password") or None),
        ),
        client_id=mqtt_cfg.get("client_id"),
    )
    publisher.start()
    return publisher


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)

    if args.list_devices:
        list_devices()
        return

    config = load_config(config_path)
    setup_logging(config.get("logging", {}))
    device_id = config.get("device_id", "linux-mic-01")

    audio_cfg = config.get("audio", {})
    sample_rate = int(audio_cfg.get("sample_rate", 16000))
    window_seconds = float(audio_cfg.get("window_seconds", 1.0))
    hop_seconds = float(audio_cfg.get("hop_seconds", 0.5))

    audio_provider = AudioStreamProvider(
        AudioStreamConfig(
            sample_rate=sample_rate,
            channels=int(audio_cfg.get("channels", 1)),
            window_seconds=window_seconds,
            hop_seconds=hop_seconds,
            mic_device_index=audio_cfg.get("mic_device_index"),
        )
    )

    capture_manager = configure_capture(config, sample_rate)
    mqtt_publisher = build_mqtt(config, args.dry_run)

    smoother_cfg = config.get("smoothing", {})
    smoother = EventSmoother(
        SmootherConfig(
            window_count=int(smoother_cfg.get("window_count", 3)),
            positives_required=int(smoother_cfg.get("positives_required", 2)),
            cooldown_seconds=float(smoother_cfg.get("cooldown_seconds", 20)),
        )
    )

    detector, detector_name, heuristic_detector, yamnet_threshold = build_detectors(
        config, sample_rate
    )

    window_samples = int(round(window_seconds * sample_rate))
    hop_samples = int(round(hop_seconds * sample_rate))
    buffer = np.zeros(0, dtype=np.float32)

    try:
        for chunk in audio_provider.stream_chunks():
            timestamp = time.time()
            completed_paths = capture_manager.extend(chunk)
            for path in completed_paths:
                logger.info("Capture finalised at {}", path)

            buffer = np.concatenate([buffer, chunk])
            while buffer.size >= window_samples:
                window = buffer[:window_samples]
                buffer = buffer[hop_samples:]

                try:
                    if detector_name == "yamnet":
                        score = float(detector.score_bark(window))
                        positive = score >= yamnet_threshold
                    else:
                        score, metrics = heuristic_detector.evaluate(window)
                        positive = heuristic_detector.is_positive(metrics)
                except Exception as exc:  # pragma: no cover - runtime safeguard
                    logger.error("Detector error ({}): {}", detector_name, exc)
                    if detector_name == "yamnet":
                        logger.warning("Switching to heuristic detector due to errors")
                        detector = heuristic_detector
                        detector_name = "heuristic"
                        continue
                    else:
                        continue

                triggered = smoother.update(positive, timestamp)
                logger.debug(
                    "Window score {:.3f} positive={} detector={}",
                    score,
                    positive,
                    detector_name,
                )

                if triggered:
                    capture_path = capture_manager.schedule_capture(timestamp, device_id)
                    payload = {
                        "event": "dog_bark",
                        "score": float(round(score, 4)),
                        "ts": int(timestamp),
                        "device_id": device_id,
                        "detector": detector_name,
                    }
                    logger.info(
                        "Bark event triggered score={:.3f} detector={} capture={}",
                        score,
                        detector_name,
                        capture_path,
                    )
                    if mqtt_publisher:
                        mqtt_publisher.publish(payload)
    except KeyboardInterrupt:
        logger.info("Interrupted by user, shutting down")
    finally:
        audio_provider.stop()
        if mqtt_publisher:
            mqtt_publisher.stop()


if __name__ == "__main__":
    main()
