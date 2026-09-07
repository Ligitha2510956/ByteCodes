import os
import re
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

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

    # --- Class weights to fix "always genuine" bias from imbalance ---
    train_labels = [labels[i] for i in train_idx]
    n_real = train_labels.count(0)
    n_fake = train_labels.count(1)
    total = n_real + n_fake
    weight_real = total / (2 * n_real)
    weight_fake = total / (2 * n_fake)
    class_weights = torch.tensor([weight_real, weight_fake], dtype=torch.float32)
    print(f"Class counts -> real: {n_real}, fake: {n_fake}, weights: {class_weights.tolist()}")

    model = DeepfakeCNN()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
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

    # --- Evaluation with full metrics ---
    model.eval()
    all_preds = []
    all_labels = []
    all_probs_fake = []  # probability of class 1 (fake) — needed for ROC-AUC

    with torch.no_grad():
        for mels, labels_batch in test_loader:
            outputs = model(mels)
            probs = F.softmax(outputs, dim=1)
            predicted = torch.argmax(outputs, dim=1)

            all_preds.extend(predicted.tolist())
            all_labels.extend(labels_batch.tolist())
            all_probs_fake.extend(probs[:, 1].tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs_fake)
    except ValueError:
        auc = float("nan")  # only one class present in test set

    cm = confusion_matrix(all_labels, all_preds)

    print(f"Test Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1 Score:       {f1:.4f}")
    print(f"ROC-AUC:        {auc:.4f}")
    print(f"Confusion Matrix (rows=true, cols=pred, 0=real,1=fake):\n{cm}")
    print(f"(Train samples: {len(train_idx)}, Test samples: {len(test_idx)}, "
          f"grouped by {len(unique_bases)} unique base clips)")

    torch.save(model.state_dict(), "deepfake_model.pth")
    print("Model saved as deepfake_model.pth")

if __name__ == "__main__":
    train_model()