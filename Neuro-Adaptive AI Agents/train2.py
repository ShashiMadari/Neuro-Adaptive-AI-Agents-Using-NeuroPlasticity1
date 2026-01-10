'''import pyttsx3

def text_to_wav(text, output_file="output.wav"):
    engine = pyttsx3.init()

    # Optional: adjust speed
    engine.setProperty('rate', 170)

    # Optional: adjust volume
    engine.setProperty('volume', 1.0)

    # Optional: choose voice (0 = male, 1 = female depending on system)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)  # change index to switch voices

    # Save as WAV
    engine.save_to_file(text, output_file)
    engine.runAndWait()

    print(f"WAV file saved as: {output_file}")


# Example use
text = """A neuro-adaptive system is an intelligent system that adapts 
its behavior based on a user's brain signals, physiological state, or 
cognitive load, adjusting responses automatically."""
text_to_wav(text, "neuro_adaptive.wav")'''

import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

models = genai.list_models()

for m in models:
    print(m.name)

