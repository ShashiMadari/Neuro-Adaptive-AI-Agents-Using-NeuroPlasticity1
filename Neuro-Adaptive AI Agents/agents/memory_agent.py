import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from reasoning.gemini_flash_reasoner import get_flash_reasoner
from sentence_transformers import SentenceTransformer # type: ignore
import torch
from torchvision import transforms
from PIL import Image
import librosa # type: ignore


class LessHearableAgent:
    def __init__(self, txt_enc, memory_store: Optional[List[Dict[str, Any]]] = None,
                 device=None, api_key: str = None,
                 memory_file: str = "memory_store/working_memory.json"):
        self.txt_enc = txt_enc
        self.device = device
        self.memory_file = memory_file
        self.flash_reasoner = get_flash_reasoner(api_key)
        self.caption_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.memory_store: List[Dict[str, Any]] = memory_store if memory_store is not None else []

        # Load existing memory
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.memory_store = data
                    else:
                        print(f"⚠️ Memory file format invalid, expected list. Loaded empty list instead.")
                print(f"✅ Loaded {len(self.memory_store)} memory items from {self.memory_file}")
            except Exception as e:
                print(f"⚠️ Could not load memory file: {e}")

    def _persist_memory(self):
        """Persist memory_store (list) to JSON file"""
        try:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(self.memory_store, f, ensure_ascii=False, indent=2)
            print(f"💾 Memory persisted ({len(self.memory_store)} items)")
        except Exception as e:
            print(f"⚠️ Failed to persist memory: {e}")

    def summarize_reasoning(self, reasoning_traces: List[str]) -> str:
        """Use GeminiFlashReasoner to generate caption/summary"""
        return self.flash_reasoner.summarize_reasoning(reasoning_traces)

    def summarize_and_store(self, query_key, reasoned_sentence, value_vec,
                            sim_threshold: float = 0.4, max_age_seconds: int = 3600):
        """
        Store minimal memory: key, value (embedding), caption, timestamp
        Then run cleanup (similarity + timestamp)
        Returns: (record, summary) for agent use
        """
        summary = self.summarize_reasoning([reasoned_sentence])

        # encode caption into embedding
        caption_embedding = self.caption_encoder.encode(summary, convert_to_numpy=True)
        caption_embedding = caption_embedding.flatten().tolist()

        record = {
            "key": query_key,
            "value": caption_embedding,
            "caption": summary,
            "timestamp": time.time()
        }

        if not isinstance(self.memory_store, list):
            self.memory_store = list(self.memory_store.values()) if isinstance(self.memory_store, dict) else []

        self.memory_store.append(record)
        self._persist_memory()
        print(f"✅ Stored memory: {reasoned_sentence[:100]}...")

        # After storing → cleanup based on new query + caption
        self.cleanup_memories(query_key, summary,
                            threshold=sim_threshold,
                            max_age_seconds=max_age_seconds)

        # ✅ Return both the memory record and the generated summary
        return record, summary


    def cleanup_memories(self, query_key: str, caption: str,
                         threshold: float = 0.4, max_age_seconds: int = 3600) -> int:
        """
        Delete memories that are BOTH:
        - Similarity (to query_key or caption) < threshold
        - Older than `max_age_seconds`
        """
        now = time.time()
        query_embedding = self.caption_encoder.encode(str(query_key), convert_to_numpy=True).reshape(1, -1)
        caption_embedding = self.caption_encoder.encode(caption, convert_to_numpy=True).reshape(1, -1)

        kept, deleted = [], 0

        for mem in self.memory_store:
            if "value" not in mem:
                continue

            mem_emb = np.array(mem["value"]).reshape(1, -1)
            sim_query = cosine_similarity(query_embedding, mem_emb)[0][0]
            sim_caption = cosine_similarity(caption_embedding, mem_emb)[0][0]
            age = now - mem.get("timestamp", now)

            # Only delete if BOTH (low similarity wrt query & caption) AND too old
            if sim_query < threshold and sim_caption < threshold or age > max_age_seconds:
                deleted += 1
                continue

            kept.append(mem)

        self.memory_store = kept
        self._persist_memory()
        print(f"🧹 Cleanup complete → Deleted {deleted}, Kept {len(self.memory_store)}.")
        return deleted
