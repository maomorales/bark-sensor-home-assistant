#!/usr/bin/env python3
"""Display real-time audio levels to help tune detection thresholds."""

import numpy as np
import sounddevice as sd
import time

SAMPLE_RATE = 16000
DURATION = 0.5  # Check every 0.5 seconds

def calculate_metrics(audio):
    """Calculate the same metrics used by the heuristic detector."""
    rms = np.sqrt(np.mean(audio ** 2))

    # Calculate band energy (400-3000 Hz)
    fft = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1.0 / SAMPLE_RATE)
    band_mask = (freqs >= 400) & (freqs <= 3000)
    band_energy = np.sum(np.abs(fft[band_mask]) ** 2)

    return rms, band_energy

print("🎤 Real-time Audio Level Monitor")
print("=" * 60)
print("This shows what the detector 'hears'")
print(f"Current thresholds in config:")
print(f"  - RMS threshold: 0.08")
print(f"  - Band energy minimum: 1.0e-4")
print("\nPress Ctrl+C to stop\n")

try:
    while True:
        # Record a short chunk
        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        audio = audio.flatten()

        rms, band_energy = calculate_metrics(audio)

        # Show if it would trigger detection
        rms_pass = rms >= 0.08
        energy_pass = band_energy >= 1.0e-4
        both_pass = rms_pass and energy_pass

        status = "🔴 WOULD DETECT" if both_pass else "⚪ Silent"

        print(f"{status} | RMS: {rms:.4f} {'✓' if rms_pass else '✗'} | "
              f"Band Energy: {band_energy:.2e} {'✓' if energy_pass else '✗'}")

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n\n👋 Stopped")
