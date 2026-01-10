import numpy as np
from typing import List, Dict, Any
from utils.similarity import cosine_sim
from utils.preprocessing import keyword_overlap
from config.settings import ALPHA, KEYWORD_BOOST, MODALITY_BOOST # Assuming these settings exist

class MemoryRetriever:
    """
    Handles retrieval of memories based on multimodal query inputs.
    """
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
    
    def retrieve_topn(self, query_list: List[Dict[str, Any]], top_n: int = 5, alpha: float = ALPHA) -> List[Dict[str, Any]]:
        """
        Retrieves the top N memories by combining scores from all modalities in the query list.

        Args:
            query_list (List[Dict[str, Any]]): A list of query dictionaries, one for each modality.
            top_n (int): The number of top memories to return.
            alpha (float): The weighting factor for key (encoder) vs. value (semantic) similarity.

        Returns:
            List[Dict[str, Any]]: The top N memories, sorted by combined score.
        """
        memory = self.memory_manager.load_memory()
        
        # List to store all calculated scores (one entry per memory item, per modality)
        all_scores = [] 

        # 1. Iterate over each individual modality query
        for query_entry in query_list:
            qtext = query_entry.get("query_text", "")
            qtype = query_entry.get("type", "")
            
            # Ensure required keys exist before proceeding
            if "value" not in query_entry or "key" not in query_entry:
                continue

            # 2. Score this modality query against all memory items
            for item in memory:
                # Calculate core similarities
                sv = cosine_sim(query_entry["value"], item["value"])
                sk = cosine_sim(query_entry["key"], item["key"])
                
                # Calculate keyword overlap (only using the query text from the current modality)
                overlap = keyword_overlap(qtext, item.get("caption", ""))
                
                # Base score: weighted average of key and value similarities
                base = (1 - alpha) * sv + alpha * sk
                
                # Bonus calculation
                modal = item.get("type", "")
                
                # Check for modality match (qtype is the type of the current query entry)
                modality_match_bonus = MODALITY_BOOST.get(modal, 0.0) * (1.0 if modal == qtype else 0.0)
                
                # Total score
                score = base + (KEYWORD_BOOST * overlap) + modality_match_bonus
                
                # Append score result
                all_scores.append({
                    # Identifier for the memory item
                    "memory_id": item.get("id", "N/A"), 
                    "memory_type": modal,
                    # Source modality that triggered this match
                    "source_modality": qtype, 
                    "caption": item.get("caption", ""),
                    "score": float(score),
                    "sim_value": float(sv),
                    "sim_key": float(sk),
                    "overlap": float(overlap),
                    "meta": item.get("processor", {})
                })
        
        # 3. Combine and filter
        # NOTE: A memory item might appear multiple times (once for image query, once for text query).
        # We sort by score and take the top unique matches.
        
        # Sort by score in descending order
        all_scores.sort(key=lambda x: x["score"], reverse=True)

        # Simple deduplication (optional, but good practice if memory retrieval is expensive/redundant)
        # For simplicity in this demo, we skip explicit deduplication and just return the top N highest scores
        # which means one memory item might appear multiple times if it scored highly against different modalities.
        
        return all_scores[:top_n]
