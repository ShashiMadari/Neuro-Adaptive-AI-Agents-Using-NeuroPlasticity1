# reasoning/semantic_utils.py
import re
from typing import List

def extract_semantic_concepts(text: str) -> List[str]:
    """Extracts key semantic concepts from text"""
    stopwords = {"a", "an", "the", "is", "are", "in", "on", "at", "and", "or", "but"}
    words = [w for w in text.lower().split() if w not in stopwords and len(w) > 2]
    return words

def extract_context_elements(text: str) -> List[str]:
    """Extract context elements from text"""
    context = []
    locations = ["park", "street", "room", "kitchen", "beach", "sea", "ocean", 
                "garden", "office", "bench", "grass", "water", "indoors", "outdoors"]
    actions = ["sitting", "standing", "playing", "running", "walking", 
              "lying", "sleeping", "eating", "drinking"]
    
    text_lower = text.lower()
    context.extend(loc for loc in locations if loc in text_lower)
    context.extend(action for action in actions if action in text_lower)
    
    return context if context else ["general context"]