# ----------------------------------
# output_layer.py
# ----------------------------------
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
import os

class OutputLayer:
    def __init__(self, facts_path=None, knowledge_path=None, threshold=0.4):
        """
        Initialize OutputLayer with precomputed embeddings.
        facts_path: path to .pt file storing fact embeddings
        knowledge_path: path to .pt file storing knowledge embeddings
        """
        self.threshold = threshold
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Load facts if path provided
        if facts_path:
            self._load_facts(facts_path)
        else:
            self.fact_keys, self.fact_texts, self.fact_emb = [], [], np.array([])

        # Load knowledge if path provided
        if knowledge_path:
            self._load_knowledge(knowledge_path)
        else:
            self.know_keys, self.know_items, self.know_emb = [], [], np.array([])

    # -------------------------------
    # Load facts from .pt
    # -------------------------------
    def _load_facts(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Facts embeddings file not found: {path}")
        data = torch.load(path)
        self.fact_keys = data.get("keys", [])
        self.fact_texts = data.get("texts", [])
        self.fact_emb = data.get("embeddings", torch.tensor([])).cpu().numpy()

    # -------------------------------
    # Load knowledge from .pt
    # -------------------------------
    def _load_knowledge(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Knowledge embeddings file not found: {path}")
        data = torch.load(path)
        self.know_keys = data.get("know_keys", [])
        self.know_items = data.get("know_items", [])
        self.know_emb = data.get("know_emb", torch.tensor([])).cpu().numpy()

    # -------------------------------
    # Retrieve top facts using semantic similarity
    # -------------------------------
    def retrieve_facts(self, query, top_k=3):
        query_emb = self.model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
        scores = util.cos_sim(query_emb, torch.tensor(self.fact_emb))
        best_idx = torch.topk(scores[0], k=top_k).indices.tolist()
        return [(self.fact_keys[i], self.fact_texts[i], float(scores[0][i])) for i in best_idx]

    # -------------------------------
    # Search combined facts + knowledge
    # -------------------------------
    def search(self, query, top_n=3):
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]

        # Cosine similarity for facts
        fact_scores = np.dot(self.fact_emb, q_emb) if self.fact_emb.size > 0 else np.array([])

        # Cosine similarity for knowledge
        know_scores = np.dot(self.know_emb, q_emb) if self.know_emb.size > 0 else np.array([])

        # Top indices
        fact_idx = np.argsort(-fact_scores)[:top_n] if fact_scores.size > 0 else []
        know_idx = np.argsort(-know_scores)[:top_n] if know_scores.size > 0 else []

        # Thresholded results
        facts_res = [{self.fact_keys[i]: self.fact_texts[i]} for i in fact_idx if fact_scores[i] > self.threshold]
        know_res = [{self.know_keys[i]: self.know_items[i]} for i in know_idx if know_scores[i] > self.threshold]

        return {"facts": facts_res, "knowledge": know_res}

    # -------------------------------
    # Pretty formatting
    # -------------------------------
    def format_output(self, query, result):
        output_lines = [f"\n🔍 Query: {query}"]

        # Facts
        output_lines.append("👉 Facts:")
        if result["facts"]:
            for f in result["facts"]:
                for k, v in f.items():
                    output_lines.append(f"   • {k}: {v}")
        else:
            output_lines.append("   • None")

        # Knowledge
        output_lines.append("👉 Knowledge:")
        if result["knowledge"]:
            for k in result["knowledge"]:
                for name, item in k.items():
                    output_lines.append(f"   • {name}")
                    if isinstance(item, dict):
                        if "statement" in item:
                            output_lines.append(f"      ↪ Statement: {item['statement']}")
                        if "formula" in item:
                            output_lines.append(f"      ↪ Formula: {item['formula']}")
                        if "explanation" in item:
                            output_lines.append(f"      ↪ Explanation: {item['explanation']}")
                        if "example" in item:
                            output_lines.append(f"      ↪ Example: {item['example']}")
                    else:
                        output_lines.append(f"      ↪ {item}")
        else:
            output_lines.append("   • None")

        return "\n".join(output_lines)

    # -------------------------------
    # Full pipeline
    # -------------------------------
    def run(self, query, top_n=3):
        result = self.search(query, top_n=top_n)
        pretty = self.format_output(query, result)
        return result, pretty


# -------------------------------
# Example usage
# -------------------------------
'''if __name__ == "__main__":
    layer = OutputLayer(
        facts_path="knowledge/real_world_facts.pt",
        knowledge_path="knowledge/knowledge_embeddings.pt"
    )
    queries = ["energy mass relation", "force formula", "triangle rule", "gravity"]
    for q in queries:
        res, pretty = layer.run(q, top_n=3)
        print(pretty)'''
