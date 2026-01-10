import numpy as np

def cosine_sim(a, b) -> float:
    a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))