# utils/projection_utils.py

import numpy as np
import os

# Set projection size
PROJECTION_DIM = 128
PROJECTION_FILE = "data/down_projection.npy"

# Load projection matrix if exists, else create
if os.path.exists(PROJECTION_FILE):
    W = np.load(PROJECTION_FILE)
else:
    # Dynamically create W based on embedding model dimension
    from utils.embedding_model import get_embedding_dim
    EMBEDDING_DIM = get_embedding_dim()
    np.random.seed(42)
    W = np.random.randn(PROJECTION_DIM, EMBEDDING_DIM).astype(np.float32)
    np.save(PROJECTION_FILE, W)

def f_down(embedding):
    """Project D-dimensional embedding → 128-d"""
    if isinstance(embedding, list):
        embedding = np.array(embedding)
    
    if embedding.shape[0] != W.shape[1]:
        print(f"[ERROR] Dimension mismatch: Expected {W.shape[1]}, got {embedding.shape[0]}")
        return np.zeros(PROJECTION_DIM, dtype=np.float32)
    
    return (W @ embedding).astype(np.float32)

def f_up(embedding_128):
    """Project 128-d → original D-d"""
    if isinstance(embedding_128, list):
        embedding_128 = np.array(embedding_128)
    return (W.T @ embedding_128).astype(np.float32)

def get_projection_shape():
    """Return shape of projection matrix"""
    return W.shape  # (128, D)
