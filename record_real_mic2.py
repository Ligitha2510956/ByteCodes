import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000
DURATION = 3

speaker_name = input("Enter a short name/label for this speaker (e.g. mom, dad, friend1): ").strip()
num_clips = int(input("How many clips to record for this speaker? (e.g. 8): "))

os.makedirs("data/real", exist_ok=True)

print(f"\nRecording {num_clips} clips for '{speaker_name}'. Speak naturally, different sentences each time.\n")
time.sleep(2)

for i in range(num_clips):
    print(f"[{i+1}/{num_clips}] Recording... speak now!")
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    audio = recording.flatten()

    fname = f"data/real/mic_{speaker_name}_{i:03d}.wav"
    sf.write(fname, audio, SAMPLE_RATE)
    print(f"    Saved {fname}")
    time.sleep(0.5)

print(f"\nDone! Recorded {num_clips} samples for '{speaker_name}'.")