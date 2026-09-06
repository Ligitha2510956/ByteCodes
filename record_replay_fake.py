import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000
DURATION = 3
NUM_CLIPS = 12

os.makedirs("data/fake", exist_ok=True)

print(f"We'll record {NUM_CLIPS} clips of REPLAYED fake audio.")
print("For each one: play a fake/cloned audio file on your laptop speakers RIGHT NOW,")
print("this script will record your mic picking it up.\n")
time.sleep(2)

for i in range(NUM_CLIPS):
    print(f"[{i+1}/{NUM_CLIPS}] Recording in 2 seconds... START PLAYING your fake audio file NOW!")
    time.sleep(2)
    print("    Recording...")
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    audio = recording.flatten()

    fname = f"data/fake/replay_fake_{i:03d}.wav"
    sf.write(fname, audio, SAMPLE_RATE)
    print(f"    Saved {fname}\n")
    time.sleep(0.5)

print(f"\nDone! Recorded {NUM_CLIPS} replayed-fake samples.")