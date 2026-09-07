"""
preprocess_all.py
-------------------
Master data preprocessing pipeline for the Voice Deepfake Detection project.

Regenerates the full data/real and data/fake dataset from public sources.
Run this once before train.py if setting up the project fresh.

NOTE: Personal recordings (your own mic samples, F5-TTS voice clones) are
NOT reproducible by this script since they involve manual recording/generation.
Those are documented separately below.
"""

import subprocess
import sys

STEPS = [
    ("get_real_data.py",     "Downloading REAL speech - Kathbath (Hindi)"),
    ("get_fake_data.py",     "Downloading FAKE speech - IndicSynth (Hindi)"),
    ("get_real_tamil.py",    "Downloading REAL speech - Kathbath (Tamil)"),
    ("get_fake_tamil.py",    "Downloading FAKE speech - IndicSynth (Tamil)"),
    ("get_itw_balanced.py",  "Downloading REAL+FAKE - In-the-Wild benchmark (English)"),
    ("augment_real.py",      "Augmenting REAL data - noise/gain variation"),
    ("augment_fake.py",      "Augmenting FAKE data - noise/gain variation"),
]

def run_step(script, description):
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"Running: {script}")
    print('='*60)
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"\n⚠️  {script} exited with an error. Check the output above.")
        print("You can fix the issue and rerun this script - already-downloaded files won't be redone.")
    else:
        print(f"✅ {description} - done")

if __name__ == "__main__":
    print("Starting full data preprocessing pipeline...")
    print("This will populate data/real/ and data/fake/ from public datasets.\n")

    for script, description in STEPS:
        run_step(script, description)

    print("\n" + "="*60)
    print("✅ Automated preprocessing complete.")
    print("="*60)
    print("""
NOTE - Manual steps NOT covered by this script (personal/generated data):
  1. Live mic recordings of real speakers (yourself, teammates, family)
     -> Run: python record_real_mic.py
     -> Run: python record_real_mic2.py (for additional named speakers)

  2. AI voice clone samples (F5-TTS or similar), added to data/fake/
     -> Generated manually via a voice-cloning tool, then moved into data/fake/

  3. Replayed-fake samples (fake audio played through speaker + re-recorded via mic)
     -> Run: python record_replay_fake4.py

These manual steps add real-world microphone/replay-channel realism that
purely synthetic/dataset-sourced audio can't fully replicate.

Once all data is ready, run: python train.py
""")