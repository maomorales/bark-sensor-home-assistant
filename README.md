# Bark Detector PoC

Python-based proof-of-concept that detects dog barks from a live Linux microphone and publishes MQTT events for Home Assistant automations. By default it uses the YAMNet TFLite model and falls back to a heuristic detector if the model cannot be loaded.

This program not only detects dog barks in real time, but also publishes these events over MQTT so that you can easily create automations in Home Assistant—for example, turning on lights, starting a recording on a camera, or activating a smart pet feeder when barking is detected.

Additionally, bark events can be sent to Dailybot, allowing you to build other types of automations such as real-time notifications in Slack or Discord, or alerting specific people whenever barking occurs.


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

### DailyBot Integration

You can configure BarkDetector to send notifications to a [DailyBot](https://dailybot.com/) workflow when a bark is detected.

**Requirements:**
- A DailyBot account.
- A workflow with a **Trigger type: Universal API** (see [DailyBot docs on Universal Triggers](https://help.dailybot.com)).
- Your workflow's trigger URL.

**Steps:**

1. **Create a new workflow in DailyBot**  
   Go to your DailyBot dashboard and create a new workflow.  
   Set the trigger type to **Universal API**.

2. **Configure fields for the trigger:**  
   - Set the event type to `"hardware_sensor"`.  
   - Set the secret `"sensor"` (or another value, but then update the code's `send_dailybot_event()` to match).
   - Optionally, configure any additional fields in the workflow for your use.

3. **Copy the Workflow Trigger URL:**  
   In the workflow's settings, you'll see a _Universal API Trigger URL_ (it looks like `https://api.dailybot.com/integrations/event/UUID-here/`).

4. **Update your config:**  
   In your `config.yaml` (or override at runtime with `--dailybot`), set the workflow URL:
   ```yaml
   dailybot:
     workflow_url: "https://api.dailybot.com/integrations/event/UUID-here/"
   ```
   This URL will be used for POST requests when the detector triggers a bark event.

**Payload details**  
When a bark event is detected, the following JSON payload is sent to DailyBot:
- `event_type` (default: `"hardware_sensor"`)
- `secret` (default: `"sensor"`)
- All bark detection metadata (`event`, `score`, `ts`, `device_id`, `detector`, etc.)
- `capture_path` (if audio capture is enabled)

If you use a custom `event_type` or `secret`, make sure to also update `main.py` in the `send_dailybot_event` function, so the payload matches what your workflow expects.

**Enabling DailyBot integration**  
- Use the `--dailybot` flag with `main.py` to enable sending events:
  ```bash
  python3 main.py --config config/config.yaml --dailybot
  ```
- If `workflow_url` is missing from config, DailyBot events will _not_ be sent.

For troubleshooting, check logs to verify POSTs are reaching DailyBot.
