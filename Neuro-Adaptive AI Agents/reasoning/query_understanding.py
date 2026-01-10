import re
from enum import Enum
from typing import Dict, Any

class QueryType(Enum):
    WHO = "who"
    WHAT = "what"
    WHERE = "where"
    YESNO = "yesno"
    EMOTION = "emotion"
    STATEMENT = "statement"
    COMPARISON = "comparison"
    REFERENCE = "reference"

_Q_EMO = {
    "happy": "joy", "happiness": "joy", "joy": "joy",
    "sad": "sadness", "sadness": "sadness",
    "angry": "anger", "anger": "anger", "mad": "anger",
    "fear": "fear", "scared": "fear", "afraid": "fear",
    "surprise": "surprise", "surprised": "surprise",
    "disgust": "disgust", "disgusted": "disgust"
}

def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", s.lower()).strip()

def classify_query(qtext: str) -> Dict[str, Any]:
    qn = norm(qtext)
    q = qn.strip()
    
    if "same" in qn or "again" in qn:
        return {"type": QueryType.COMPARISON.value, "emotion": None}
    if "this" in qn or "that" in qn:
        return {"type": QueryType.REFERENCE.value, "emotion": None}
    
    qtype = "statement"
    if q.startswith("who ") or " who " in q:
        qtype = "who"
    elif q.startswith("where ") or " where " in q:
        qtype = "where"
    elif q.startswith("what ") or " what " in q:
        qtype = "what"
    elif q.endswith("?") or any(q.startswith(k) for k in ["is ", "are ", "does ", "do ", "did ", "can ", "has ", "have "]):
        qtype = "yesno"
    
    emo = None
    for k, v in _Q_EMO.items():
        if re.search(rf"\b{k}\b", q):
            emo = v
            break
    
    if "emotion" in q or "sentiment" in q:
        m = re.search(r"emotion\s*[:\-]\s*([a-z]+)", q)
        if m:
            emo = _Q_EMO.get(m.group(1), m.group(1))
        qtype = "emotion"
    
    if emo and qtype == "statement":
        qtype = "emotion"
    
    return {"type": qtype, "emotion": emo}

def extract_subject(text: str) -> str:
    """Extract main subject from text with better handling"""
    if not text:
        return "unknown"
    
    text_lower = text.lower()
    
    # Common subjects to look for
    subjects = [
        "dog", "cat", "person", "man", "woman", "girl", "boy", "child",
        "car", "tree", "house", "building", "animal", "bird", "flower"
    ]
    
    # Check for specific subjects
    for subject in subjects:
        if subject in text_lower:
            return subject
    
    # Handle comparison queries
    if "same" in text_lower or "again" in text_lower:
        return "subject"  # Generic for comparisons
    
    # Extract first noun or important word
    words = text_lower.split()
    if words:
        return words[0]
    
    return "unknown"