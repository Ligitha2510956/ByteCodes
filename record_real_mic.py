import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000
DURATION = 3
NUM_CLIPS = 15

os.makedirs("data/real", exist_ok=True)

print(f"We'll record {NUM_CLIPS} short clips of you speaking, {DURATION} sec each.")
print("Speak naturally, different sentences each time. Get ready...\n")
time.sleep(2)

for i in range(NUM_CLIPS):
    print(f"[{i+1}/{NUM_CLIPS}] Recording... speak now!")
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    audio = recording.flatten()

    fname = f"data/real/mic_real_{i:03d}.wav"
    sf.write(fname, audio, SAMPLE_RATE)
    print(f"    Saved {fname}")
    time.sleep(0.5)

print("\nDone! Recorded", NUM_CLIPS, "real mic samples.")