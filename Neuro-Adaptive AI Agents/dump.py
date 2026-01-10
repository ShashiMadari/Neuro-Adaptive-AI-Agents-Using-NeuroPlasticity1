import pyttsx3

# Initialize the engine
engine = pyttsx3.init()

# Your joke text
joke_text = "What does the Law of Conservation of Energy mean?"

# Save as WAV file
engine.save_to_file(joke_text, "blind3.wav")

# Run the engine to process the audio
engine.runAndWait()

print("Audio saved as joke_audio.wav")
