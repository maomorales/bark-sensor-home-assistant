#!/usr/bin/env python3
"""Debug script to see what YAMNet classifies in real-time audio."""

import numpy as np
import sounddevice as sd
import time
from detector.yamnet import YAMNetBarkDetector, YAMNetConfig

SAMPLE_RATE = 16000
DURATION = 0.384  # Same as config

print("🤖 YAMNet Real-time Classification Debug")
print("=" * 70)
print("This shows ALL sounds YAMNet detects, not just barks")
print("Press Ctrl+C to stop\n")

# Initialize YAMNet
config = YAMNetConfig(
    model_url="https://huggingface.co/qualcomm/YamNet/resolve/main/YamNet_float.tflite",
    classes_url="https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv",
    conf_threshold=0.01,
    label_substrings=("dog", "bark", "bow", "yip"),
)

try:
    detector = YAMNetBarkDetector(config)
    print("✅ YAMNet loaded successfully\n")
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

        # Get ALL top predictions (access internal method)
        scores, _ = detector.interpreter.invoke(audio)
        top_indices = np.argsort(scores.mean(axis=0))[::-1][:5]

        print(f"\n🎯 Bark Score: {bark_score:.4f} {'🔴 BARK!' if bark_score >= 0.01 else ''}")
        print("   Top 5 classifications:")
        for i, idx in enumerate(top_indices, 1):
            class_name = detector.class_names.get(idx, f"Unknown({idx})")
            score = scores.mean(axis=0)[idx]
            is_bark = any(s in class_name.lower() for s in ["dog", "bark", "bow", "yip"])
            marker = "🐶" if is_bark else "  "
            print(f"   {marker} {i}. {class_name:30s} ({score:.4f})")

        time.sleep(0.5)

except KeyboardInterrupt:
    print("\n\n👋 Stopped")
