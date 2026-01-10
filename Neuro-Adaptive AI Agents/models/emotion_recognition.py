from transformers import pipeline

# Load once
emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)

def detect_text_emotion(text):
    result_list = emotion_model(text)
    
    # Normalize the output
    if isinstance(result_list, list) and isinstance(result_list[0], list):
        result = result_list[0][0]
    elif isinstance(result_list, list):
        result = result_list[0]
    else:
        raise ValueError("Unexpected output format from emotion model")

    emotion = result['label']
    score = result['score']

    # Assign importance based on score
    if score > 0.8:
        importance = 1.0
    elif score > 0.5:
        importance = 0.7
    else:
        importance = 0.4

    return emotion, score, importance
