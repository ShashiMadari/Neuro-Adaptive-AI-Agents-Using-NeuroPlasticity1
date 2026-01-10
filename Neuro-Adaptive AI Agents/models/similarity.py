import torch
from sentence_transformers import util
from utils.embedding_model import embed_text
from utils.projection_utils import f_down

def compute_similarity(input_text, kb_texts, kb_embeddings, input_embedding=None):
    from utils.embedding_model import embed_text
    import torch
    from sentence_transformers import util

    if input_embedding is not None:
        query_vec = input_embedding
    else:
        query_vec = embed_text(input_text)

    kb_tensor = torch.tensor(kb_embeddings, dtype=torch.float32)
    input_tensor = torch.tensor(query_vec, dtype=torch.float32).unsqueeze(0)
    similarities = util.cos_sim(input_tensor, kb_tensor)[0]  # Shape: (N,)

    best_idx = torch.argmax(similarities).item()
    return kb_texts[best_idx], similarities[best_idx].item()

