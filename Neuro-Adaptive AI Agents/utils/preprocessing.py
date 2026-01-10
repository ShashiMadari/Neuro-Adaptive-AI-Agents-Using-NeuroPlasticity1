import re
import math
from typing import List

# Stopwords for QUERY ONLY
STOP_QUERY = {
    "a","an","the","in","on","at","to","with","is","are","was","were","out","of","and","or",
    "this","that","these","those","for","from","by","into","as","it","its","her","his","their",
    "who","what","where","when","why","how","same"
}

def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", s.lower()).strip()

def tokens_query(s: str) -> List[str]:
    return [w for w in norm(s).split() if w and w not in STOP_QUERY]

def tokens_doc(s: str) -> List[str]:
    return [w for w in norm(s).split() if w]

def keyword_overlap(q: str, caption: str) -> float:
    qset, cset = set(tokens_query(q)), set(tokens_doc(caption))
    if not qset or not cset:
        return 0.0
    inter = len(qset & cset)
    return inter / math.sqrt(len(qset) * len(cset))