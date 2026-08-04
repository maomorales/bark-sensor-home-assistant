#!/usr/bin/env python3
"""Test YAMNet on a captured WAV file."""

import sys
import wave
import numpy as np
from detector.yamnet import YAMNetBarkDetector, YAMNetConfig

if len(sys.argv) < 2:
    print("Usage: python3 test_yamnet_file.py <wav_file>")
    sys.exit(1)

wav_file = sys.argv[1]

# Load YAMNet
config = YAMNetConfig(
    model_url="https://huggingface.co/qualcomm/YamNet/resolve/main/YamNet_float.tflite",
    classes_url="https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv",
    conf_threshold=0.01,
    label_substrings=("dog", "bark", "bow", "yip"),
)

print("Loading YAMNet...")
detector = YAMNetBarkDetector(config)
print(f"✅ YAMNet loaded")
print(f"   Bark indices found: {detector._bark_indices}")
print(f"   Total classes: {len(detector._bark_indices)}")

# Load WAV file
print(f"\nLoading WAV file: {wav_file}")
with wave.open(wav_file, 'rb') as wf:
    sample_rate = wf.getframerate()
    n_frames = wf.getnframes()
    audio_bytes = wf.readframes(n_frames)
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

print(f"   Sample rate: {sample_rate} Hz")
print(f"   Duration: {len(audio) / sample_rate:.2f} seconds")
print(f"   Audio range: [{audio.min():.3f}, {audio.max():.3f}]")

# Test in chunks
window_size = 6144  # ~0.384s at 16kHz
hop_size = 3072

print(f"\nProcessing audio in chunks...")
max_score = 0.0
for i in range(0, len(audio) - window_size, hop_size):
    chunk = audio[i:i+window_size]
    score = detector.score_bark(chunk)
    if score > max_score:
        max_score = score
    if score > 0.0:
        print(f"   Time {i/sample_rate:.2f}s: bark score = {score:.4f}")

print(f"\n📊 Maximum bark score: {max_score:.4f}")
if max_score >= 0.01:
    print("✅ BARK DETECTED!")
else:
    print("❌ No barks detected (all scores were 0.000)")
