import subprocess
import torch
import imageio_ffmpeg
from train import audio_to_melspec, DeepfakeCNN

def convert_mp3_to_wav(mp3_path):
    wav_path = mp3_path.replace(".mp3", "_converted.wav")
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg_exe, "-y", "-i", mp3_path, wav_path], check=True)
    return wav_path

# Load the trained model
model = DeepfakeCNN()

dummy_mel = audio_to_melspec("data/real/real_0000.wav")
dummy_tensor = torch.tensor(dummy_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
model(dummy_tensor)

model.load_state_dict(torch.load("deepfake_model.pth"))
model.eval()

def predict(path):
    if path.endswith(".mp3"):
        path = convert_mp3_to_wav(path)

    mel = audio_to_melspec(path)
    mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        output = model(mel_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred = torch.argmax(probs).item()
    label = "FAKE" if pred == 1 else "REAL"
    confidence = probs[pred].item() * 100
    print(f"{path} -> Predicted: {label}  ({confidence:.2f}% confidence)")

test_files = [
    "my_voice_tamil.mp3",
    "test_fake.wav",
     "my_voice_tamil2.mp3",
    "my_voice_cloned.wav",
]

for f in test_files:
    predict(f)
