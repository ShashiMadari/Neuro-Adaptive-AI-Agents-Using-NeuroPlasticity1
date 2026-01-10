import json
import joblib
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import librosa

# --- Import processors ---
from processors.image_processor import process_image
from processors.audio_processor import process_audio
from processors.text_processor import process_text

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

# ---------- COMPUTE 5D KEYS FROM RAW INPUT ----------
def get_5d_key_from_raw(image_path=None, audio_path=None, text_input=None, modality=None):
    if modality == "image":
        img = Image.open(image_path).convert("RGB")
        img = img.resize((224, 224))
        img_arr = np.array(img).reshape(1, -1)
        return pca_image.transform(img_arr)
    elif modality == "audio":
        y, sr = librosa.load(audio_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc, axis=1).reshape(1, -1)
        return pca_audio.transform(mfcc_mean)
    elif modality == "text":
        text_embed = text_model.encode([text_input])
        return pca_text.transform(text_embed)
    else:
        raise ValueError("Unknown modality")

# ---------- EMBED CAPTIONS ----------
def embed_caption(text):
    return text_model.encode([text])

# ---------- SIMILARITY CHECK ----------
def check_across_modalities(input_key, input_caption, wm_data, input_modality="input", cross_modal=False):
    matches = []
    input_cap_embed = embed_caption(input_caption)

    for modality_name, entries in wm_data.items():
        if cross_modal and modality_name == input_modality:
            continue
        for entry in entries:
            stored_key = np.array(entry["key_5d"]).reshape(1, -1)
            key_sim = cosine_similarity(input_key, stored_key)[0][0]

            if not cross_modal and key_sim < IMAGE_SIM_THRESHOLD:
                continue

            stored_cap_embed = np.array(entry["caption_384d"]).reshape(1, -1)
            cap_sim = cosine_similarity(input_cap_embed, stored_cap_embed)[0][0]

            if cap_sim >= CAPTION_SIM_THRESHOLD:
                matches.append({
                    "input_modality": input_modality,
                    "matched_modality": modality_name,
                    "caption": entry.get("caption", ""),
                    "emotion": entry.get("emotion", ""),
                    "key_similarity": float(key_sim),
                    "caption_similarity": float(cap_sim)
                })
    return matches

# ---------- MAIN PIPELINE ----------
def main(image_path, audio_path, text_input):
    print("\n🚀 Processing Inputs with Processors...")

    # --- Get processor outputs ---
    img_result = process_image(image_path)
    aud_result = process_audio(audio_path)
    txt_result = process_text(text_input)

    print("\n🖼️ Image Output:")
    print(" Emotion:", img_result['emotion'])
    print(" Caption:", img_result['caption'])
    print(" Importance:", img_result['importance'])

    print("\n🔊 Audio Output:")
    print(" Transcribed:", aud_result['transcribed'])
    print(" Emotion:", aud_result['emotion'])
    print(" Importance:", aud_result['importance'])

    print("\n📝 Text Output:")
    print(" Summary:", txt_result['text_summary'])
    print(" Emotion:", txt_result['emotion'])
    print(" Importance:", txt_result['importance'])

    # --- Compute 5D keys ---
    img_key_5d = get_5d_key_from_raw(image_path=image_path, modality="image")
    aud_key_5d = get_5d_key_from_raw(audio_path=audio_path, modality="audio")
    txt_key_5d = get_5d_key_from_raw(text_input=text_input, modality="text")

    # --- Prepare captions for similarity ---
    img_caption = img_result.get("caption", "image input")
    aud_caption = aud_result.get("transcribed", "audio input")
    txt_caption = text_input

    wm_data = {
        "image": image_memory,
        "audio": audio_memory,
        "text": text_memory
    }

    # --- Within-modality matches ---
    img_matches = check_across_modalities(img_key_5d, img_caption, {"image": image_memory}, input_modality="image")
    aud_matches = check_across_modalities(aud_key_5d, aud_caption, {"audio": audio_memory}, input_modality="audio")
    txt_matches = check_across_modalities(txt_key_5d, txt_caption, {"text": text_memory}, input_modality="text")

    # --- Cross-modality matches ---
    img_cross = check_across_modalities(img_key_5d, img_caption, wm_data, input_modality="image", cross_modal=True)
    aud_cross = check_across_modalities(aud_key_5d, aud_caption, wm_data, input_modality="audio", cross_modal=True)
    txt_cross = check_across_modalities(txt_key_5d, txt_caption, wm_data, input_modality="text", cross_modal=True)

    all_matches = img_matches + aud_matches + txt_matches + img_cross + aud_cross + txt_cross

    # --- Deduplicate & sort ---
    seen = set()
    unique_matches = []
    for m in all_matches:
        key = (m['matched_modality'], m['caption'])
        if key not in seen:
            seen.add(key)
            unique_matches.append(m)
    unique_matches.sort(key=lambda x: x['caption_similarity'], reverse=True)

    print("\n✅ Matches Found:")
    for m in unique_matches:
        print(m)

# ---------- RUN ----------
if __name__ == "__main__":
    image_path = "data/WhatsApp Image 2025-07-30 at 15.15.58_6b2d361c.jpg"
    audio_path = "data/audio.wav"
    text_input = "What is color of flower"

    main(image_path, audio_path, text_input)
