import os
import numpy as np
import librosa
import soundfile as sf

SAMPLE_RATE = 16000
SOURCE_DIR = "data/real"

def add_noise(audio, noise_level=0.005):
    noise = np.random.normal(0, noise_level, audio.shape)
    return audio + noise

def change_gain(audio, factor):
    return audio * factor

files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".wav") and not f.startswith("aug_")]

count = 0
for fname in files:
    path = os.path.join(SOURCE_DIR, fname)
    audio, sr = librosa.load(path, sr=SAMPLE_RATE)

    # Version 1: add background noise (simulates non-studio mic)
    noisy = add_noise(audio, noise_level=0.008)
    sf.write(os.path.join(SOURCE_DIR, f"aug_noise_{fname}"), noisy, sr)

    # Version 2: lower volume + slight noise (simulates phone/laptop mic)
    quiet_noisy = add_noise(change_gain(audio, 0.6), noise_level=0.005)
    sf.write(os.path.join(SOURCE_DIR, f"aug_quiet_{fname}"), quiet_noisy, sr)

    count += 1

print(f"Created {count * 2} augmented REAL samples in {SOURCE_DIR}")