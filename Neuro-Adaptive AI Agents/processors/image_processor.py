# processors/image_processor.py

from deepface import DeepFace # type: ignore
from models.image_captioning import generate_caption

def process_image(image_path):
    result = {}

    try:
        analysis = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)
        emotion = analysis[0]['dominant_emotion']
        result['emotion'] = emotion
        result['face_detected'] = True

        # Assign importance based on emotion category
        strong_emotions = ['angry', 'fear', 'sad', 'happy']
        neutral_emotions = ['neutral', 'surprise', 'disgust']

        if emotion in strong_emotions:
            importance = 1.0
        elif emotion in neutral_emotions:
            importance = 0.6
        else:
            importance = 0.4  # Unknown or less meaningful

    except:
        result['emotion'] = 'none'
        result['face_detected'] = False
        importance = 0.3  # Low importance if no face

    result['importance'] = importance
    result['caption'] = generate_caption(image_path)
    result['image_text'] = f"Emotion: {result['emotion']}, Caption: {result['caption']}"
    
    return result
