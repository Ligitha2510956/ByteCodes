import os
import re
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

SAMPLE_RATE = 16000
DURATION = 3
N_MELS = 64

def array_to_melspec(audio):
    # Normalize loudness so mic volume/gain doesn't affect the result
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    target_len = SAMPLE_RATE * DURATION
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def audio_to_melspec(path):
    audio, sr = librosa.load(path, sr=SAMPLE_RATE)
    return array_to_melspec(audio)

def get_base_id(fname):
    """
    Strips augmentation prefixes so that an original file and its
    augmented siblings (noise/quiet versions) are recognized as the
    same underlying recording. This lets us group them together before
    train/test splitting, so augmented copies of a clip never end up
    on the opposite side of the split from their original -- which
    would otherwise leak information between train and test.
    """
    name = fname
    name = re.sub(r'^aug_(noise|quiet)_', '', name)
    return name

class VoiceDataset(Dataset):
    def __init__(self, real_dir="data/real", fake_dir="data/fake"):
        self.samples = []
        self.labels = []

        for fname in os.listdir(real_dir):
            self.samples.append(os.path.join(real_dir, fname))
            self.labels.append(0)

        for fname in os.listdir(fake_dir):
            self.samples.append(os.path.join(fake_dir, fname))
            self.labels.append(1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mel = audio_to_melspec(self.samples[idx])
        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return mel, label

class DeepfakeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

def train_model():
    torch.manual_seed(42)  # makes every run reproducible

    dataset = VoiceDataset()
    labels = dataset.labels

    # --- Grouped train/test split to prevent data leakage ---
    base_ids = [get_base_id(os.path.basename(p)) for p in dataset.samples]

    base_to_label = {}
    for base, label in zip(base_ids, labels):
        base_to_label[base] = label

    unique_bases = list(base_to_label.keys())
    unique_labels = [base_to_label[b] for b in unique_bases]

    train_bases, test_bases = train_test_split(
        unique_bases, test_size=0.2, stratify=unique_labels, random_state=42
    )
    train_bases = set(train_bases)
    test_bases = set(test_bases)

    train_idx = [i for i, b in enumerate(base_ids) if b in train_bases]
    test_idx = [i for i, b in enumerate(base_ids) if b in test_bases]

    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=8, shuffle=True)
    test_loader = DataLoader(torch.utils.data.Subset(dataset, test_idx), batch_size=8)

    model = DeepfakeCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    EPOCHS = 10
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for mels, labels_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(mels)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for mels, labels_batch in test_loader:
            outputs = model(mels)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels_batch).sum().item()
            total += labels_batch.size(0)

    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    print(f"(Train samples: {len(train_idx)}, Test samples: {len(test_idx)}, "
          f"grouped by {len(unique_bases)} unique base clips)")

    torch.save(model.state_dict(), "deepfake_model.pth")
    print("Model saved as deepfake_model.pth")

if __name__ == "__main__":
    train_model()