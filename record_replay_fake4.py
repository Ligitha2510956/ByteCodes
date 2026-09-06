import os
import random
import time
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000
DURATION = 3

FAKE_DIR = "data/fake"

existing_fake_files = [
    f for f in os.listdir(FAKE_DIR)
    if f.endswith(".wav") and not f.startswith(("aug_", "replay_", "replay2_", "replay3_", "replay4_"))
]

random.shuffle(existing_fake_files)

num_clips = int(input(f"How many replay-fake clips to record? (max {len(existing_fake_files)}): "))
selected = existing_fake_files[:num_clips]

print("\nThe script will automatically play each fake clip through your speakers AND record")
print("your mic picking it up, perfectly in sync. You just press Enter to move to the next one.\n")

count = 0
for fname in selected:
    input(f"[{count+1}/{len(selected)}] Press Enter to play + record this clip (source: {fname})...")

    path = os.path.join(FAKE_DIR, fname)
    audio, sr = librosa.load(path, sr=SAMPLE_RATE)

    target_len = SAMPLE_RATE * DURATION
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    # Fade in/out over 50ms to avoid clicking/popping sounds at start and end
    fade_len = int(0.05 * SAMPLE_RATE)
    fade_in = np.linspace(0, 1, fade_len)
    fade_out = np.linspace(1, 0, fade_len)
    audio[:fade_len] *= fade_in
    audio[-fade_len:] *= fade_out

    recorded = sd.playrec(audio.reshape(-1, 1), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    recorded = recorded.flatten()

    out_fname = f"data/fake/replay4_{count:03d}.wav"
    sf.write(out_fname, recorded, SAMPLE_RATE)
    print(f"    Saved {out_fname}\n")
    count += 1

print(f"\nDone! Created {count} synced replayed-fake samples.")