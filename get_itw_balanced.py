import os
import soundfile as sf
from datasets import load_dataset

dataset = load_dataset("SpeechAntiSpoofingBenchmarks/InTheWild", split="test", streaming=True)

os.makedirs("data/real", exist_ok=True)
os.makedirs("data/fake", exist_ok=True)

MAX_EACH = 60

real_count = 0
fake_count = 0

for example in dataset:
    if real_count >= MAX_EACH and fake_count >= MAX_EACH:
        break

    label = example.get("label")  # 1 = spoof/fake, 0 = bonafide/real

    try:
        audio = example["audio"]["array"]
        sr = example["audio"]["sampling_rate"]

        if label == 0 and real_count < MAX_EACH:
            sf.write(f"data/real/itw_real_{real_count:04d}.wav", audio, sr)
            real_count += 1
        elif label == 1 and fake_count < MAX_EACH:
            sf.write(f"data/fake/itw_clone_{fake_count:04d}.wav", audio, sr)
            fake_count += 1
    except Exception as e:
        print(f"Skipping a sample due to error: {e}")

print(f"Saved {real_count} In-the-Wild REAL samples and {fake_count} In-the-Wild FAKE samples")