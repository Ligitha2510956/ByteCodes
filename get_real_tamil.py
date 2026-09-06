import os
import soundfile as sf
from datasets import load_dataset

dataset = load_dataset("ai4bharat/Kathbath", "tamil", split="train", streaming=True)

os.makedirs("data/real", exist_ok=True)

count = 0
for example in dataset:
    if count >= 200:
        break
    audio = example["audio_filepath"]["array"]
    sr = example["audio_filepath"]["sampling_rate"]
    sf.write(f"data/real/real_ta_{count:04d}.wav", audio, sr)
    count += 1

print(f"Saved {count} Tamil REAL samples to data/real/")