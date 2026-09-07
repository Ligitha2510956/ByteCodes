import threading
import numpy as np
import sounddevice as sd
import torch
import tkinter as tk
from collections import deque
from train import array_to_melspec, DeepfakeCNN, SAMPLE_RATE, DURATION, N_MELS

model = DeepfakeCNN()
dummy_tensor = torch.zeros(1, 1, N_MELS, 94)
model(dummy_tensor)
model.load_state_dict(torch.load("deepfake_model.pth"))
model.eval()

WINDOW_SIZE = 6
history = deque(maxlen=WINDOW_SIZE)
chunk_history = deque(maxlen=10)
running = True

BG = "#0f1117"
CARD = "#1a1d29"
ACCENT = "#3b82f6"
GREEN = "#2ecc71"
YELLOW = "#f5b642"
RED = "#e74c3c"
GRAY = "#3a3f4d"
TEXT_MUTED = "#8b8fa3"

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

def audio_loop():
    global running
    while running:
        recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        audio = recording.flatten()

        if np.abs(audio).mean() < 0.001:
            root.after(0, lambda: set_state("listening", 0, "Waiting for speech..."))
            continue

        label, confidence = predict_chunk(audio)
        fake_score = confidence / 100 if label == "FAKE" else 1 - (confidence / 100)
        history.append(fake_score)
        chunk_history.append(fake_score)
        smoothed = sum(history) / len(history)
        smoothed_conf = smoothed * 100

        if smoothed >= 0.55:
            root.after(0, lambda c=smoothed_conf: set_state("fake", c, f"Chunk: {label} {confidence:.0f}%"))
        elif smoothed <= 0.35:
            root.after(0, lambda c=smoothed_conf: set_state("real", c, f"Chunk: {label} {confidence:.0f}%"))
        else:
            root.after(0, lambda c=smoothed_conf: set_state("suspicious", c, f"Chunk: {label} {confidence:.0f}%"))

def set_state(state, fake_pct, subtitle):
    if state == "listening":
        color, status = GRAY, "LISTENING"
    elif state == "real":
        color, status = GREEN, "GENUINE"
    elif state == "suspicious":
        color, status = YELLOW, "SUSPICIOUS"
    else:
        color, status = RED, "DEEPFAKE"

    canvas.itemconfig(circle, fill=color, outline=color)
    canvas.itemconfig(glow, outline=color)
    status_label.config(text=status, fg=color)
    subtitle_label.config(text=subtitle)

    bar_width = int(360 * (fake_pct / 100))
    bar_canvas.coords(bar_fill, 0, 0, bar_width, 18)
    bar_canvas.itemconfig(bar_fill, fill=color)
    pct_label.config(text=f"{fake_pct:.1f}% fake-likelihood")

    draw_history()

def draw_history():
    dot_canvas.delete("all")
    x = 10
    for score in chunk_history:
        if score >= 0.55:
            c = RED
        elif score <= 0.35:
            c = GREEN
        else:
            c = YELLOW
        dot_canvas.create_oval(x, 4, x + 20, 24, fill=c, outline="")
        x += 28

def on_close():
    global running
    running = False
    root.destroy()

root = tk.Tk()
root.title("Voice Authenticity Monitor")
root.geometry("480x620")
root.minsize(480, 620)
root.configure(bg=BG)

top_bar = tk.Frame(root, bg=ACCENT, height=6)
top_bar.pack(fill="x", side="top")

tk.Label(root, text="🎙️ VOICE AUTHENTICITY MONITOR", font=("Segoe UI", 13, "bold"),
         bg=BG, fg="white").pack(pady=(28, 4))
tk.Label(root, text="Real-time AI voice cloning detection", font=("Segoe UI", 9),
         bg=BG, fg=TEXT_MUTED).pack(pady=(0, 20))

canvas = tk.Canvas(root, width=220, height=220, bg=BG, highlightthickness=0)
canvas.pack(pady=5)
glow = canvas.create_oval(5, 5, 215, 215, fill="", outline=GRAY, width=3)
circle = canvas.create_oval(25, 25, 195, 195, fill=GRAY, outline=GRAY, width=0)

status_label = tk.Label(root, text="STARTING", font=("Segoe UI", 28, "bold"), bg=BG, fg="white")
status_label.pack(pady=(20, 4))

subtitle_label = tk.Label(root, text="Initializing model...", font=("Segoe UI", 11), bg=BG, fg=TEXT_MUTED)
subtitle_label.pack(pady=(0, 25))

bar_bg = tk.Frame(root, bg=CARD, width=380, height=26)
bar_bg.pack(pady=5)
bar_bg.pack_propagate(False)
bar_canvas = tk.Canvas(bar_bg, width=360, height=18, bg=CARD, highlightthickness=0)
bar_canvas.pack(padx=10, pady=4)
bar_fill = bar_canvas.create_rectangle(0, 0, 0, 18, fill=GRAY, outline="")

pct_label = tk.Label(root, text="0.0% fake-likelihood", font=("Segoe UI", 11), bg=BG, fg=TEXT_MUTED)
pct_label.pack(pady=(8, 30))

tk.Label(root, text="RECENT CHUNKS", font=("Segoe UI", 9, "bold"), bg=BG, fg=TEXT_MUTED).pack()
dot_canvas = tk.Canvas(root, width=300, height=30, bg=BG, highlightthickness=0)
dot_canvas.pack(pady=(8, 20))

root.protocol("WM_DELETE_WINDOW", on_close)

thread = threading.Thread(target=audio_loop, daemon=True)
thread.start()

root.mainloop()