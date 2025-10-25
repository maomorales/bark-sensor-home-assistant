# Bark Detector PoC

Python-based proof-of-concept that detects dog barks from a live Linux microphone and publishes MQTT events for Home Assistant automations. By default it uses the YAMNet TFLite model and falls back to a heuristic detector if the model cannot be loaded.

## Features
- Live audio capture via ALSA (`sounddevice`) with 16 kHz mono resampling.
- YAMNet (TFLite) inference with automatic model download on first run.
- Heuristic fallback using RMS and band-limited energy.
- Sliding window scoring (1.0 s window, 0.5 s hop) with majority vote smoothing and cooldown.
- MQTT publishing with automatic reconnection.
- Optional WAV capture around events using a rolling 20 s buffer.
- Systemd unit file and Home Assistant automation example.

## Requirements
- Python 3.10+
- ALSA-compatible microphone (built-in or USB)
- Access to an MQTT broker

Dependencies are listed in `requirements.txt` and installed via `scripts/setup.sh`.

## Quick Start

### 1. Setup
```bash
git clone https://example.com/barkdetector.git
cd barkdetector
./scripts/setup.sh
```

This creates a virtual environment, installs dependencies, and sets up directories.

### 2. Activate Virtual Environment
```bash
source .venv/bin/activate
```

### 3. Configure
Edit `config/config.yaml` to suit your environment:
- **MQTT settings**: Update `host`, `port`, `username`, `password` to match your Home Assistant MQTT broker
- **Audio device**: Use `--list-devices` (below) to find your microphone index
- **Detection mode**: Choose `yamnet` (ML-based) or `heuristic` (simpler, faster)

### 4. List Available Microphones
```bash
python3 main.py --list-devices
```

### 5. Test MQTT Connection (Optional)
Before running the detector, verify MQTT connectivity:
```bash
python3 test_mqtt.py
```
This subscribes to the bark topic and displays any incoming events. Press Ctrl+C to stop.

### 6. Run the Detector
```bash
# With MQTT publishing
python3 main.py --config config/config.yaml

# Dry run (no MQTT, testing only)
python3 main.py --config config/config.yaml --dry-run
```

On the first run, the script downloads `models/yamnet.tflite` and `models/yamnet_class_map.csv` if YAMNet mode is enabled.

## Configuration
- `config/config.yaml` contains runtime settings (audio, detection, MQTT, logging).
- `config/config.example.yaml` shows default values.
- `capture.out_dir` and `logging.file_path` default to `/var/lib/barkdetector/captures` and `/var/log/barkdetector/barkdetector.log`. Ensure the process has permission to write to these paths or adjust them locally.

## Systemd Deployment
```bash
sudo mkdir -p /opt/barkdetector
sudo cp -r . /opt/barkdetector/
sudo cp service/barkdetector.service /etc/systemd/system/barkdetector@${USER}.service
sudo systemctl daemon-reload
sudo systemctl enable barkdetector@${USER}
sudo systemctl start barkdetector@${USER}
```

The unit uses `/opt/barkdetector/config/config.yaml` as its configuration file.

## Home Assistant Automation
Import `ha_automation_example.yaml` to add a notification that fires when the MQTT topic receives a bark event.

## Notes
- When YAMNet cannot be loaded (missing model, download failure, etc.), the heuristic detector remains active so events are still produced.
- Use `--dry-run` to evaluate detection without publishing MQTT messages.
- WAV captures are disabled automatically if the configured directory is not writable.
