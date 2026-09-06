import os
import soundfile as sf
from datasets import load_dataset

dataset = load_dataset("SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF", split="test", streaming=True)

os.makedirs("data/real", exist_ok=True)
os.makedirs("data/fake", exist_ok=True)

MAX_REAL = 300
MAX_FAKE = 300

real_count = 0
fake_count = 0

for example in dataset:
    if real_count >= MAX_REAL and fake_count >= MAX_FAKE:
        break

    audio = example["audio"]["array"]
    sr = example["audio"]["sampling_rate"]
    label = example["label"]  # 0 = bonafide (real), 1 = spoof (fake)

    if label == 0 and real_count < MAX_REAL:
        sf.write(f"data/real/asvspoof_real_{real_count:04d}.wav", audio, sr)
        real_count += 1
    elif label == 1 and fake_count < MAX_FAKE:
        sf.write(f"data/fake/asvspoof_fake_{fake_count:04d}.wav", audio, sr)
        fake_count += 1

print(f"Saved {real_count} ASVspoof REAL samples and {fake_count} ASVspoof FAKE samples")