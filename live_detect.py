import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from collections import deque
from train import array_to_melspec, DeepfakeCNN, SAMPLE_RATE, DURATION, N_MELS

model = DeepfakeCNN()

dummy_tensor = torch.zeros(1, 1, N_MELS, 94)  # placeholder shape to trigger LazyLinear
model(dummy_tensor)

model.load_state_dict(torch.load("deepfake_model.pth"))
model.eval()

def predict_chunk(audio):
    mel = array_to_melspec(audio)
    mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        output = model(mel_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred = torch.argmax(probs).item()
    label = "FAKE" if pred == 1 else "REAL"
    confidence = probs[pred].item() * 100
    return label, confidence

WINDOW_SIZE = 5  # how many recent chunks to average over for the session-level verdict
history = deque(maxlen=WINDOW_SIZE)

print("🎙️  Listening... speak into your microphone. Press Ctrl+C to stop.\n")

try:
    while True:
        recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        audio = recording.flatten()

        # Skip near-silent chunks (avoid false predictions on background noise)
        if np.abs(audio).mean() < 0.001:
            print("... (silence, waiting for speech)")
            continue

        label, confidence = predict_chunk(audio)

        # Convert this chunk's result into a 0-1 "fake likelihood" score
        fake_score = confidence / 100 if label == "FAKE" else 1 - (confidence / 100)
        history.append(fake_score)

        smoothed_fake_score = sum(history) / len(history)
        smoothed_confidence = smoothed_fake_score * 100

        if smoothed_fake_score >= 0.45:
            risk = "🔴 Deepfake"
        elif smoothed_fake_score <= 0.25:
            risk = "🟢 Genuine"
        else:
            risk = "🟡 Suspicious"

        print(f"[chunk: {label} {confidence:.1f}%]  ->  SESSION: {risk}  ({smoothed_confidence:.1f}% fake-likelihood, over last {len(history)} chunks)")

except KeyboardInterrupt:
    print("\nStopped listening.")