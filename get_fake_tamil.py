import os
import soundfile as sf
from datasets import load_dataset

dataset = load_dataset("vdivyasharma/IndicSynth", name="Tamil", split="train", streaming=True)

os.makedirs("data/fake", exist_ok=True)

count = 0
for example in dataset:
    if count >= 200:
        break
    audio = example["audio"]["array"]
    sr = example["audio"]["sampling_rate"]
    sf.write(f"data/fake/fake_ta_{count:04d}.wav", audio, sr)
    count += 1

print(f"Saved {count} Tamil FAKE samples to data/fake/")