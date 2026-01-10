import os
import random
import numpy as np
import json
from tqdm import tqdm
from sklearn.decomposition import PCA
import joblib

from processors.text_processor import embed_text
from processors.image_processor import process_image
from processors.audio_processor import process_audio

# --- 1. Setup Paths ---
DATASET_PATH = r"C:\Users\shash\OneDrive\Desktop\Main-project\dataset_triplets"
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# --- 2. Load & Validate Triplet Folders ---
def find_triplet_folders(dataset_path):
    triplet_folders = []
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if not os.path.isdir(folder_path):
            continue

        files = os.listdir(folder_path)
        has_image = any(f.endswith(('.jpg', '.png')) for f in files)
        has_audio = any(f.endswith('.wav') for f in files)
        has_text = any(f.endswith('.txt') for f in files)

        if has_image and has_audio and has_text:
            triplet_folders.append(folder_path)
        else:
            print(f"❌ Skipping folder {folder_name}: Missing file(s)")

    print(f"✅ Found {len(triplet_folders)} valid triplet folders.")
    return triplet_folders

triplet_folders = find_triplet_folders(DATASET_PATH)

# --- 3. Extract Embeddings ---
embeddings_384 = []
sources = []

for folder_path in tqdm(triplet_folders[:44000]):
    try:
        files = os.listdir(folder_path)
        txt_file = [f for f in files if f.endswith('.txt')][0]
        img_file = [f for f in files if f.endswith(('.jpg', '.png'))][0]
        aud_file = [f for f in files if f.endswith('.wav')][0]

        txt_path = os.path.join(folder_path, txt_file)
        img_path = os.path.join(folder_path, img_file)
        aud_path = os.path.join(folder_path, aud_file)

        with open(txt_path, "r", encoding="utf-8") as f:
            text_data = f.read().strip()

        text_emb = embed_text(text_data)
        img_emb = process_image(img_path)['embedding']
        aud_emb = process_audio(aud_path)['embedding']

        fused_384 = (np.array(text_emb) + np.array(img_emb) + np.array(aud_emb)) / 3.0

        embeddings_384.append(fused_384)
        sources.append({
            "text": text_data,
            "image": img_path,
            "audio": aud_path,
            "id": os.path.basename(folder_path)
        })

    except Exception as e:
        print(f"⚠️ Skipped {folder_path} due to error: {e}")
        continue

print(f"✅ Extracted embeddings for {len(embeddings_384)} samples.")

# --- 4. Sample 1000 for Working Memory ---
if len(embeddings_384) < 1000:
    raise ValueError(f"Not enough data to sample 1000 working memory points. Only got {len(embeddings_384)}")

random.seed(42)
selected_idx = random.sample(range(len(embeddings_384)), 1000)

wm_384 = [embeddings_384[i] for i in selected_idx]
wm_sources = [sources[i] for i in selected_idx]

# --- 5. Train PCA 384 → 32 ---
pca = PCA(n_components=32)
pca.fit(embeddings_384)
wm_32 = pca.transform(wm_384)

# --- 6. Save PCA Model ---
joblib.dump(pca, "models/pca_384to32.joblib")

# --- 7. Save Working Memory JSON ---
working_memory = []
for i in range(1000):
    working_memory.append({
        "id": wm_sources[i]["id"],
        "embedding_384": wm_384[i].tolist(),
        "embedding_32": wm_32[i].tolist(),
        "modality": "fused",
        "source": {
            "text": wm_sources[i]["text"],
            "image": wm_sources[i]["image"],
            "audio": wm_sources[i]["audio"]
        }
    })

with open("data/working_memory.json", "w", encoding="utf-8") as f:
    json.dump(working_memory, f, indent=2)

print("💾 Working memory saved to: data/working_memory.json")
