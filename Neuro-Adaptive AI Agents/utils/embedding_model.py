# utils/embedding_model.py

from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim output

def embed_text(text: str):
    
    """Return embedding as numpy array"""
    embedding = model.encode(text, normalize_embeddings=True)
    return np.array(embedding, dtype=np.float32)

def embed_batch(text_list):
    return model.encode(text_list, normalize_embeddings=True)

def get_embedding_dim():
    return model.get_sentence_embedding_dimension()
