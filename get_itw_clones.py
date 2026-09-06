import os
import soundfile as sf
from datasets import load_dataset

dataset = load_dataset("SpeechAntiSpoofingBenchmarks/InTheWild", split="test", streaming=True)

os.makedirs("data/fake", exist_ok=True)

MAX_CLONES = 60

count = 0
for example in dataset:
    if count >= MAX_CLONES:
        break

    label = example.get("label")
    if label != 1:  # 1 = spoof/fake, 0 = bonafide/real
        continue

    try:
        audio = example["audio"]["array"]
        sr = example["audio"]["sampling_rate"]
        sf.write(f"data/fake/itw_clone_{count:04d}.wav", audio, sr)
        count += 1
    except Exception as e:
        print(f"Skipping a sample due to error: {e}")

print(f"Saved {count} In-the-Wild spoof samples to data/fake/")