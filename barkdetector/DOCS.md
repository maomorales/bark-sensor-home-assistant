# Bark Detector

Listens to a USB microphone plugged into your Home Assistant machine, classifies
audio with Google's YAMNet model, and publishes a bark event to MQTT.

Unlike camera-based sound detection (Frigate, yamcam), this reads a **locally
attached microphone**, so it works in rooms with no camera.

## Requirements

- Home Assistant OS or Supervised, on aarch64 or amd64
- A USB microphone plugged into the host
- The **Mosquitto broker** add-on (or any MQTT broker reachable from the host)

A Raspberry Pi 4 runs this easily. Measured on a Pi 4 with Home Assistant OS
18.2: **1.3% CPU and 88 MB RAM**, using the XNNPACK delegate at a 0.5 s hop.

## Installation

1. Add this repository under **Settings → Add-ons → Add-on Store → ⋮ →
   Repositories**.
2. Install **Bark Detector**.
3. Plug in the USB microphone **before starting the add-on**.
4. Start it, then open the **Log** tab.

On first start the add-on downloads the YAMNet model (~14 MB) into the image's
`models/` directory if it is not already bundled.

## Picking the microphone

The log prints every input source it can see at startup:

```
Detected audio input sources:
8  alsa_input.usb-Blue_Microphones_Yeti_Nano_...analog-stereo  ...
```

**Leave `mic_device` empty on Home Assistant OS.** Audio reaches this add-on
through Home Assistant's PulseAudio plugin, and PortAudio sees only two devices
through it — `pulse` and `default` — never your microphone by name. Selecting a
specific microphone is done at the Home Assistant level, by making it the
default input:

```
ha audio default input <source-name>     # from the SSH add-on
ha audio info                            # lists source names
```

`mic_device` only applies when running the detector standalone on a normal Linux
box, where PortAudio enumerates ALSA devices directly. There, prefer part of the
device name over an index, since indices shift across reboots and USB
re-enumeration.

If no sources are listed at all, Home Assistant's audio plugin has not picked up
the microphone. Check `ha audio info` from the SSH add-on and confirm the device
appears; a reboot after plugging it in usually resolves it.

## Configuration

| Option | Default | Description |
| --- | --- | --- |
| `device_id` | `barkdetector` | Identifies this sensor in MQTT payloads and discovery |
| `detection_mode` | `yamnet` | `yamnet` (ML) or `heuristic` (RMS + band energy) |
| `conf_threshold` | `0.2` | YAMNet confidence required to call a window a bark |
| `normalize_windows` | `false` | Boost quiet windows before inference; helps distant sounds |
| `normalize_noise_floor` | `0.005` | Windows quieter than this are not boosted |
| `normalize_max_gain` | `30` | Ceiling on the boost applied to one window |
| `mic_device` | *(empty)* | Leave empty on HAOS; see "Picking the microphone" above |
| `windows_total` | `5` | Size of the sliding vote window |
| `windows_required` | `2` | Positive windows needed within it to fire an event |
| `cooldown_seconds` | `10` | Minimum gap between events |
| `mqtt_topic` | `home/sensors/dog_bark` | Topic events are published to |
| `mqtt_discovery` | `true` | Auto-create the Home Assistant binary sensor |
| `capture_enabled` | `false` | Save a WAV around each event to `/media/barkdetector` |
| `capture_normalize_peak` | `0.7` | Amplify saved clips to this peak so quiet events are audible; `0` saves raw |
| `capture_retention_hours` | `24` | Clips older than this are deleted |
| `capture_max_mb` | `512` | Hard ceiling on the clips folder; oldest go first. `0` disables |
| `log_level` | `info` | Set to `debug` to log the score of every window |

MQTT credentials are taken from the Mosquitto add-on automatically. Only set
`mqtt_host` / `mqtt_port` / `mqtt_username` / `mqtt_password` if you use an
external broker — those manual values take priority when present.

## The entity

With `mqtt_discovery` on, a `binary_sensor` named **Bark** appears under a *Bark
Detector* device. It turns on when an event is published and clears itself after
10 seconds, since the detector only reports barks and never an "all clear".

Example automation:

```yaml
automation:
  - alias: "Notify on bark"
    triggers:
      - trigger: state
        entity_id: binary_sensor.bark_detector_bark
        to: "on"
    actions:
      - action: notify.notify
        data:
          message: >-
            Dog barking detected
            (confidence {{ state_attr('binary_sensor.bark_detector_bark', 'score') }})
```

The raw MQTT payload is also available if you prefer a manual sensor:

```json
{"event": "dog_bark", "score": 0.83, "ts": 1754246400, "device_id": "barkdetector", "detector": "yamnet"}
```

### Disk usage

Each clip is ~320 KB (10 s of 16 kHz mono). Two independent limits apply, and
whichever bites first wins:

- **Age** — `capture_retention_hours` deletes clips older than the window,
  swept hourly.
- **Size** — `capture_max_mb` is a hard ceiling, checked after *every* clip is
  written. When exceeded, the oldest clips are deleted immediately even if
  they are inside the retention window.

The size limit is the one that actually protects the disk: a noisy night can
produce thousands of clips that are all younger than 24 hours, so age alone
bounds nothing. At the 512 MB default that is roughly 1,600 clips.

Clips are written to `/media/barkdetector`, so they appear under **Media → My
media → barkdetector** in the sidebar and play directly in the browser.
Filenames are `YYYYMMDD_HHMMSS_<device_id>.wav` and sort chronologically.

Distant events can be 40 dB below full scale, which is inaudible on playback
even though the detector scores them fine. `capture_normalize_peak` amplifies
each saved clip so its loudest moment reaches the given level. This touches the
saved file only -- detection always runs on the unmodified audio -- so it cannot
affect accuracy. Set it to 0 if you want the raw levels for analysis.

Log files are separately capped at 25 MB (5 MB × 5 rotations).

## Tuning

Start with `log_level: debug` and watch the per-window scores while your dog
barks.

- **Missing barks** — lower `conf_threshold` (try 0.15) or lower
  `windows_required` to 2. Small or distant dogs score lower.
- **False positives** from speech or TV — raise `conf_threshold` toward 0.4, or
  raise `windows_required`, which demands the sound persist rather than spike.
- **Events arriving in bursts** — raise `cooldown_seconds`.

`windows_required` / `windows_total` is the main precision lever, but mind the
arithmetic: with a 0.975 s window and a 0.5 s hop, any instant falls in only
**2** windows. A single short bark can therefore never satisfy
`windows_required: 3` — that setting demands ~1.5 s of sustained sound. Use 2 to
catch one-off barks, 3 to reject transients like doors and dropped cutlery.

### On distant sounds

Detection is limited by the microphone, not by these settings. A bark arriving
below the microphone's self-noise cannot be recovered by gain, thresholds or
normalisation, because all of them scale signal and noise together. If distant
barks you can clearly hear produce no rise in the `peak` value logged at debug
level, the recording genuinely does not contain them, and the fix is microphone
placement or a more sensitive/directional microphone.

## DailyBot integration (optional)

Set `dailybot_workflow_url` to a [DailyBot](https://dailybot.com/) workflow
trigger URL and each bark event is also POSTed there, for routing to Slack,
Discord or a person. The payload carries the same fields as MQTT plus
`event_type: hardware_sensor` and `secret: sensor`. Leave it empty to disable.

## Troubleshooting

**"No MQTT broker available"** — install the Mosquitto broker add-on, or set
`mqtt_host` manually.

**Log says the heuristic detector is active** — YAMNet failed to load; the line
above it says why. The add-on deliberately keeps running on the fallback
detector rather than crashing.

**No audio sources listed** — see "Picking the microphone" above. Note that
Home Assistant's audio plugin holds the ALSA devices, so another add-on reading
`/dev/snd` directly can block this one.
