<img src="barkdetector/logo.png" alt="Bark Detector" width="320">

# Bark Detector — Home Assistant add-on repository

Detect dog barks from a **USB microphone attached to your Home Assistant
machine**, and get a `binary_sensor` you can automate on.

Audio is classified locally by Google's [YAMNet][yamnet] via TensorFlow Lite.
Nothing leaves your network.

[![Open your Home Assistant instance and show the add add-on repository dialog.](https://my.home-assistant.io/badges/supervisor_add_addon_repository.svg)](https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https%3A%2F%2Fgithub.com%2Fmaomorales%2Fbark-sensor-home-assistant)

## Install

**Settings → Add-ons → Add-on Store → ⋮ → Repositories**, and add:

```
https://github.com/maomorales/bark-sensor-home-assistant
```

Then install **Bark Detector**. Full documentation is in
[`barkdetector/DOCS.md`](barkdetector/DOCS.md).

> The first install builds the image on your machine, which takes several
> minutes on a Raspberry Pi. Subsequent updates are much faster.

## How it differs from what already exists

[Frigate][frigate] and [yamcam][yamcam] classify audio from **camera RTSP
streams**, and both work well. This add-on reads a **locally attached
microphone**, for rooms with no camera or spots a camera cannot reach.

## Features

- YAMNet (TFLite) classification, with a heuristic fallback if the model fails
  to load, so detection degrades rather than stops
- MQTT discovery creates the `binary_sensor` — no manual YAML
- MQTT credentials taken from the Mosquitto add-on via the Supervisor
- Sliding-window voting with a cooldown, to reject one-off transients
- Optional clips saved to `/media`, playable in the Media panel, with the
  confidence score in the filename
- Storage bounded by both age and total size

Resource use on a Raspberry Pi 4: **1.3% CPU, 88 MB RAM**.

## Example automation

```yaml
triggers:
  - trigger: state
    entity_id: binary_sensor.bark_detector_bark
    to: "on"
actions:
  - action: notify.notify
    data:
      message: >-
        🐶 Bark detected (confidence
        {{ state_attr('binary_sensor.bark_detector_bark', 'score') }})
```

More examples, including the raw MQTT form, in
[`ha_automation_example.yaml`](ha_automation_example.yaml).

## The microphone is the hard part

If the dogs you care about are far away, **the microphone determines whether
this works** — not the software settings.

Measured on the same Pi, in the same room, with identical settings:

| | Cheap USB electret | USB condenser |
| --- | --- | --- |
| Idle noise floor (peak) | 0.0115 | 0.0055 |
| Distant barks clearly audible to a person | **nothing recorded** | detected, score 0.668 |

Cheap capsules have a self-noise floor near 35–40 dB SPL. A distant bark is not
merely quiet, it is *below that floor* — absent from the recording entirely. Gain,
thresholds and normalisation all scale signal and noise together, so none of them
recover it.

What actually helps, in order of impact:

1. **Move the microphone outdoors or to an open window** — worth 20–30 dB, more
   than every software option combined
2. **Use a low-self-noise capsule** — a condenser rather than a cheap electret
3. **Point a directional mic at the source** — roughly 6 dB, and unlike gain it
   genuinely improves the ratio by rejecting noise the signal does not share

Set expectations accordingly: dogs in your house or a neighbour's yard are
reliable. A kilometre away, outdoors, on a quiet night is best-effort.

## Running standalone

The detector also runs on any Linux box without Home Assistant. The application
lives in [`barkdetector/app/`](barkdetector/app).

```bash
git clone https://github.com/maomorales/bark-sensor-home-assistant.git
cd bark-sensor-home-assistant/barkdetector/app
./scripts/setup.sh
cp config/example-config.yaml config/config.yaml   # then edit
python3 main.py --config config/config.yaml
```

Requires Python 3.10–3.11: `tflite-runtime` publishes no wheel past CPython 3.11.
On 3.12+ install `ai-edge-litert` instead, which the code also accepts.

MQTT credentials can be supplied as `BARKDETECTOR_MQTT_HOST` / `_PORT` /
`_TOPIC` / `_USERNAME` / `_PASSWORD`, which override the config file — keep them
out of version control.

Useful flags:

```bash
python3 main.py --list-devices             # find your microphone
python3 main.py --config ... --dry-run     # detect without publishing
```

## Repository layout

```
repository.yaml          Home Assistant add-on repository manifest
barkdetector/            the add-on
  config.yaml            options schema and add-on metadata
  Dockerfile             Debian-based image (see note below)
  build.yaml             per-architecture base images
  rootfs/                s6 service and options-to-config renderer
  app/                   the Python application
```

### Why Debian and not the default Alpine base

`numpy`, `scipy` and `tflite-runtime` publish manylinux (glibc) wheels only. On
Alpine's musl, pip falls back to building from source and fails on a Pi. The base
is pinned to **bookworm** specifically, because bookworm ships Python 3.11 and
`tflite-runtime` 2.14.0 — its final release — has no wheel beyond cp311.

`numpy` is pinned below 2.0 for the same reason: `tflite-runtime` is compiled
against the NumPy 1.x C ABI, and NumPy 2.x makes the interpreter fail to load
with `_ARRAY_API not found`.

## Licence

MIT — see [LICENSE](LICENSE).

[yamnet]: https://www.tensorflow.org/hub/tutorials/yamnet
[frigate]: https://docs.frigate.video/configuration/audio_detectors/
[yamcam]: https://github.com/cecat/CeC-HA-Addons
