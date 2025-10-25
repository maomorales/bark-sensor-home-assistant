#!/usr/bin/env python3
"""Simple YAMNet bark score monitor."""

import numpy as np
import sounddevice as sd
import time
from detector.yamnet import YAMNetBarkDetector, YAMNetConfig

SAMPLE_RATE = 16000
DURATION = 0.384  # Same as config
THRESHOLD = 0.01  # Same as config

print("🤖 YAMNet Bark Score Monitor")
print("=" * 60)
print(f"Threshold: {THRESHOLD} (1%)")
print("Press Ctrl+C to stop\n")

# Initialize YAMNet
config = YAMNetConfig(
    model_url="https://huggingface.co/qualcomm/YamNet/resolve/main/YamNet_float.tflite",
    classes_url="https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv",
    conf_threshold=THRESHOLD,
    label_substrings=("dog", "bark", "bow", "yip"),
)

try:
    detector = YAMNetBarkDetector(config)
    print("✅ YAMNet loaded successfully")
    print(f"   Found {len(detector._bark_indices)} bark-related classes\n")
except Exception as e:
    print(f"❌ Failed to load YAMNet: {e}")
    exit(1)

try:
    while True:
        # Record a chunk
        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        audio = audio.flatten()

        # Get bark score
        bark_score = detector.score_bark(audio)
        would_trigger = bark_score >= THRESHOLD

        status = "🔴 BARK DETECTED!" if would_trigger else "⚪ Listening..."
        bar_len = int(bark_score * 50)
        bar = "█" * bar_len + "░" * (50 - bar_len)

        print(f"{status:20s} | Score: {bark_score:.4f} | [{bar}]")

        time.sleep(0.2)

except KeyboardInterrupt:
    print("\n\n👋 Stopped")
