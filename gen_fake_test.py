import pyttsx3

engine = pyttsx3.init()
engine.save_to_file("This is a test sentence spoken by a computer voice.", "test_fake.wav")
engine.runAndWait()
print("Saved test_fake.wav")