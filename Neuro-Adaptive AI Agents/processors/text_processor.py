# processors/text_processor.py

from transformers import pipeline  # type: ignore
from sentence_transformers import SentenceTransformer
from models.emotion_recognition import detect_text_emotion
import numpy as np

# Load models

text_model = SentenceTransformer('all-MiniLM-L6-v2')  # 384D

def process_text(input_text):
    """
    Process input text using transformer-based emotion detection.
    Returns structured results: emotion, score, importance, and a summary.
    """

    # Use the Hugging Face emotion model
    emotion, score, importance = detect_text_emotion(input_text)

    # Generate a summary line
    summary = f"Emotion: {emotion}, Text: {input_text.strip()}"

    return {
        'emotion': emotion,
        'emotion_score': round(score, 3),
        'importance': round(importance, 3),
        'matched_kb': None,
        'similarity': 0.0,
        'text_summary': summary
    }
