import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000
DURATION = 3

os.makedirs("data/fake", exist_ok=True)

num_clips = int(input("How many replay-fake clips do you want to record? (e.g. 50): "))

print("\nFor each round:")
print("1. Get your fake/cloned audio file ready to play (paused, cued up)")
print("2. Press Enter here")
print("3. Immediately click play on the audio")
print("4. Recording starts right after you press Enter and lasts 3 seconds\n")

count = 0
for i in range(num_clips):
    input(f"[{i+1}/{num_clips}] Press Enter, then immediately hit play on your fake audio...")
    print("    Recording NOW...")
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    audio = recording.flatten()

    fname = f"data/fake/replay3_{count:03d}.wav"
    sf.write(fname, audio, SAMPLE_RATE)
    print(f"    Saved {fname}\n")
    count += 1

print(f"\nDone! Recorded {count} manually-timed replayed-fake samples.")