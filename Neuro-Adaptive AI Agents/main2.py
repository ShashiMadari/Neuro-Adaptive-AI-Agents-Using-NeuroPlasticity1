import json
import joblib
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import librosa

# ---------- CONFIG ----------
IMAGE_PCA_MODEL_PATH = "models/pca_image_key_5D.joblib"
AUDIO_PCA_MODEL_PATH = "models/pca_audio_5d.joblib"
TEXT_PCA_MODEL_PATH = "long_term_memory/text_pca_model_5d.pkl"

IMAGE_LTM_PATH = "long_term_memory/image_ltm_key_value.json"
AUDIO_LTM_PATH = "long_term_memory/audio_ltm.json"
TEXT_LTM_PATH = "long_term_memory/text_ltm_with_pca.json"

IMAGE_SIM_THRESHOLD = 0.10
CAPTION_SIM_THRESHOLD = 0.10

# ---------- LOAD MODELS ----------
print("📂 Loading PCA models...")
pca_image = joblib.load(IMAGE_PCA_MODEL_PATH)
pca_audio = joblib.load(AUDIO_PCA_MODEL_PATH)
pca_text = joblib.load(TEXT_PCA_MODEL_PATH)

text_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- LOAD WORKING MEMORY ----------
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

image_memory = load_json(IMAGE_LTM_PATH)
audio_memory = load_json(AUDIO_LTM_PATH)
text_memory = load_json(TEXT_LTM_PATH)

# ---------- EMBEDDING FUNCTIONS ----------
def process_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img_arr = np.array(img).reshape(1, -1)
    return pca_image.transform(img_arr)

def process_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1).reshape(1, -1)
    return pca_audio.transform(mfcc_mean)

def process_text_key(text):
    text_embed = text_model.encode([text])
    return pca_text.transform(text_embed)

def embed_caption(text):
    return text_model.encode([text])

# ---------- SIMILARITY CHECK ----------
def check_across_modalities(input_key, input_caption, wm_data):
    matches = []
    input_cap_embed = embed_caption(input_caption)  # embed only once

    for modality_name, entries in wm_data.items():
        for entry in entries:
            stored_key = np.array(entry["key_5d"]).reshape(1, -1)
            key_sim = cosine_similarity(input_key, stored_key)[0][0]

            if key_sim < IMAGE_SIM_THRESHOLD:
                print(f"[SKIP] Low key sim {key_sim:.2f} for {entry.get('caption')}")
                continue

            stored_cap_embed = np.array(entry["caption_384d"]).reshape(1, -1)
            cap_sim = cosine_similarity(input_cap_embed, stored_cap_embed)[0][0]

            if cap_sim >= CAPTION_SIM_THRESHOLD:
                matches.append({
                    "modality": modality_name,
                    "caption": entry.get("caption", ""),
                    "emotion": entry.get("emotion", ""),
                    "key_similarity": float(key_sim),
                    "caption_similarity": float(cap_sim)
                })
    return matches

# ---------- MAIN PIPELINE ----------
def main(image_path, audio_path, text_input):
    print("\n🚀 Processing Inputs...")
    img_key = process_image(image_path)
    aud_key = process_audio(audio_path)
    txt_key = process_text_key(text_input)

    wm_data = {
        "image": image_memory,
        "audio": audio_memory,
        "text": text_memory
    }

    print("\n🔍 Matching Image Key...")
    img_matches = check_across_modalities(img_key, text_input, {"image": image_memory})

    print("\n🔍 Matching Audio Key...")
    aud_matches = check_across_modalities(aud_key, text_input, {"audio": audio_memory})

    print("\n🔍 Matching Text Key...")
    txt_matches = check_across_modalities(txt_key, text_input, {"text": text_memory})

    all_matches = img_matches + aud_matches + txt_matches

    print("\n✅ Matches Found:")
    for m in all_matches:
        print(m)

# ---------- RUN ----------
if __name__ == "__main__":
    image_path = "data/23445819_3a458716c1.jpg"
    audio_path = "data/audio.wav"
    text_input = "A dog playing in the park"

    main(image_path, audio_path, text_input)
