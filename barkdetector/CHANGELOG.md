# Changelog

## 0.3.0

- New `capture_normalize_peak` option (default 0.7): saved clips are amplified
  so quiet, distant events are audible on playback. Detection is unaffected --
  it always scores the unmodified audio.

## 0.2.1

- Write captures to `/media/barkdetector` instead of `/share`, so they are
  browsable and playable from Home Assistant's Media panel.
- Correct the `mic_device` documentation: under Home Assistant OS, PortAudio
  only ever sees `pulse` and `default`, so the microphone is chosen by setting
  the default input in Home Assistant, not by this option.

## 0.2.0

- Optional per-window normalisation before YAMNet inference, for quiet or
  distant sounds. Off by default; see `normalize_windows`.
- Debug logs now report `rms`, `peak` and applied `gain` per window, which makes
  a silent or misrouted microphone obvious.
- Audio routing diagnostics on startup when `log_level` is `debug`.
- Default `windows_required` lowered from 3 to 2: with a 0.975 s window and
  0.5 s hop a single short bark can only ever occupy 2 windows, so 3 made
  one-off barks undetectable by construction.

## 0.1.1

- Pin `numpy<2`: tflite-runtime 2.14.0 is built against the NumPy 1.x C ABI, so
  NumPy 2.x made YAMNet fail to load and silently fall back to the heuristic.
- Cap capture storage by total size as well as age, enforced after every write.
- New options: `capture_retention_hours`, `capture_max_mb`.

## 0.1.0

First release as a Home Assistant add-on.

- YAMNet bark detection from a locally attached USB microphone.
- MQTT credentials taken automatically from the Mosquitto add-on.
- MQTT discovery creates the `binary_sensor` without manual YAML.
- Microphone selectable by name, not just by unstable PortAudio index.
- Falls back to the heuristic detector instead of crashing when the TFLite
  interpreter is unavailable.
