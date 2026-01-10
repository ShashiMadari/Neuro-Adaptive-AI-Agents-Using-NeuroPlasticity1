# working_memory.py

import numpy as np
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer, util
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from models.image_captioning import generate_caption
from models.speech_to_text import transcribe_audio, convert_audio_to_vosk_format
import os

# Load sentence embedding model
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load CLIP for image feature extraction
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = clip_model.to(device)

# Working memory storage (should be loaded from memory or file)
reduced_embeddings = []  # List of 32D PCA vectors
metadata = []            # [{caption_or_text, emotion}]
pca_model = None         # To be set after initial PCA training


# --------- Feature Extraction (new input) ----------
def extract_clip_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    return features[0].cpu().numpy()


def extract_audio_embedding(audio_path):
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    import torchaudio

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)

    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0, keepdim=True)  # mono
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
    waveform = resampler(waveform).to(device)

    inputs = processor(waveform.squeeze(), return_tensors="pt", sampling_rate=16000).to(device)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings[0].cpu().numpy()


# ---------- Main Retrieval API -------------
def query_working_memory(input_path, input_type="image", similarity_threshold=0.8):
    global pca_model

    if pca_model is None:
        raise ValueError("PCA model not initialized. Run training before querying.")

    # 1. Extract feature
    if input_type == "image":
        original_embedding = extract_clip_embedding(input_path)
        caption = generate_caption(input_path)
    elif input_type == "audio":
        temp_path = "temp_converted.wav"
        convert_audio_to_vosk_format(input_path, temp_path)
        original_embedding = extract_audio_embedding(temp_path)
        caption = transcribe_audio(temp_path)
        os.remove(temp_path)
    else:
        raise ValueError("Unsupported input type")

    # 2. Reduce using existing PCA
    reduced_query = pca_model.transform([original_embedding])[0]

    # 3. Compare with working memory keys (32D)
    max_sim = -1
    best_match_idx = None
    for idx, emb in enumerate(reduced_embeddings):
        sim = cosine_similarity(reduced_query, emb)
        if sim > max_sim:
            max_sim = sim
            best_match_idx = idx

    print(f"Top PCA similarity = {max_sim:.3f}")

    if max_sim >= similarity_threshold:
        retrieved_text = metadata[best_match_idx]["caption_or_text"]

        # 4. Convert both to 384D sentence embeddings
        current_embed = sentence_model.encode(caption, convert_to_tensor=True)
        retrieved_embed = sentence_model.encode(retrieved_text, convert_to_tensor=True)

        semantic_sim = util.pytorch_cos_sim(current_embed, retrieved_embed).item()
        print(f"Semantic Similarity = {semantic_sim:.3f}")

        return {
            "input_caption": caption,
            "retrieved_caption": retrieved_text,
            "pca_similarity": float(max_sim),
            "semantic_similarity": float(semantic_sim),
            "trigger_reasoning": semantic_sim >= 0.7
        }
    else:
        print("No similar memory found.")
        return {
            "input_caption": caption,
            "retrieved_caption": None,
            "pca_similarity": float(max_sim),
            "semantic_similarity": None,
            "trigger_reasoning": False
        }


# --------- Utils --------------
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
