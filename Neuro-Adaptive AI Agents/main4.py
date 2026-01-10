import os
import json
import torch
import numpy as np
from typing import List, Dict, Any, Tuple,Optional



from processors.image_processor import process_image
from processors.audio_processor import process_audio
from processors.text_processor import process_text

from multimodal_train import ImageEncoder, AudioEncoder, TextEncoder, DEVICE
from sentence_transformers import SentenceTransformer
from torchvision import transforms
from PIL import Image
import librosa
import re
import math

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification
)

# --------------------------
# Config
# --------------------------
EMBED_DIM = 15
MEMORY_FILE = "memory_store/working_memory.json"
TOP_N = 5                         # how many items to retrieve

# 🔁 Reasoner: stronger than flan-base, still light-ish
REASONER_MODEL = "google/flan-t5-large"

# ✅ NLI model for entailment guardrails (keeps final grounded)
NLI_MODEL = "facebook/bart-large-mnli"

# Modal boosts (nudge to prefer same-modality evidence)
MODALITY_BOOST = {
    "image": 0.03,
    "text": 0.03,
    "audio": 0.02
}

# Keyword overlap weight into final retrieval score
KEYWORD_BOOST = 0.12

# Alpha for fusion of (value 384D) vs (key 15D)
ALPHA = 0.20

# --------------------------
# Load trained encoders
# --------------------------
img_enc = ImageEncoder(out_dim=EMBED_DIM).to(DEVICE)
aud_enc = AudioEncoder(out_dim=EMBED_DIM).to(DEVICE)
txt_enc = TextEncoder(out_dim=EMBED_DIM).to(DEVICE)

img_enc.load_state_dict(torch.load("models/img_encoder_15d.pt", map_location=DEVICE))
aud_enc.load_state_dict(torch.load("models/aud_encoder_15d.pt", map_location=DEVICE))
txt_enc.load_state_dict(torch.load("models/txt_encoder_15d.pt", map_location=DEVICE))

img_enc.eval(); aud_enc.eval(); txt_enc.eval()

# Semantic encoder (384D, frozen)
sem_model = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------
# Reasoner (generator) + NLI
# --------------------------
tok = AutoTokenizer.from_pretrained(REASONER_MODEL)
reasoner = AutoModelForSeq2SeqLM.from_pretrained(REASONER_MODEL).to(DEVICE)
reasoner.eval()

nli_tok = AutoTokenizer.from_pretrained(NLI_MODEL)
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(DEVICE)
nli_model.eval()

def run_reasoner(prompt: str, max_new_tokens: int = 96) -> str:
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = reasoner.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=4,
            length_penalty=0.9,
            early_stopping=True,
        )
    return tok.decode(out[0], skip_special_tokens=True).strip()

def nli_entailment_score(premise: str, hypothesis: str) -> float:
    """
    Return P(entailment) for (premise -> hypothesis).
    """
    inputs = nli_tok(
        premise, hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(DEVICE)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()
    # MNLI order: [contradiction, neutral, entailment]
    return float(probs[2])

# --------------------------
# Preprocessing utils
# --------------------------
img_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
])

def load_image(path):
    return img_tf(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)

def load_audio(path, sr=16000, n_mels=64, target_frames=128):
    y, sr = librosa.load(path, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    t = mel_db.shape[1]
    if t < target_frames:
        mel_db = np.pad(mel_db, ((0,0),(0,target_frames-t)), mode="constant")
    elif t > target_frames:
        start = (t - target_frames)//2
        mel_db = mel_db[:, start:start+target_frames]
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std()+1e-6)
    return torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

# --------------------------
# Cosine similarity
# --------------------------
def cosine_sim(a, b) -> float:
    a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

# --------------------------
# Improved Text Processing
# --------------------------

# Stopwords for QUERY ONLY (we do NOT drop 'girl', 'dog', etc. from captions)
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
    # Very light filtering for docs; keep entity nouns like 'girl', 'dog', 'bench'
    return [w for w in norm(s).split() if w]

def keyword_overlap(q: str, caption: str) -> float:
    qset, cset = set(tokens_query(q)), set(tokens_doc(caption))
    if not qset or not cset:
        return 0.0
    inter = len(qset & cset)
    return inter / math.sqrt(len(qset) * len(cset))

# --------------------------
# Query Understanding
# --------------------------
_Q_EMO = {"happy":"joy", "happiness":"joy", "joy":"joy",
          "sad":"sadness", "sadness":"sadness",
          "angry":"anger", "anger":"anger", "mad":"anger",
          "fear":"fear", "scared":"fear", "afraid":"fear",
          "surprise":"surprise", "surprised":"surprise",
          "disgust":"disgust", "disgusted":"disgust"}

def classify_query(qtext: str) -> Dict[str, Any]:
    qn = norm(qtext)
    q = qn.strip()
    qtype = "statement"
    if q.startswith("who ") or " who " in q:
        qtype = "who"
    elif q.startswith("where ") or " where " in q:
        qtype = "where"
    elif q.startswith("what ") or " what " in q:
        qtype = "what"
    elif q.endswith("?") or any(q.startswith(k) for k in ["is ", "are ", "does ", "do ", "did ", "can ", "has ", "have "]):
        qtype = "yesno"
    # detect emotion intent
    emo = None
    for k,v in _Q_EMO.items():
        if re.search(rf"\b{k}\b", q):
            emo = v
            break
    if "emotion" in q or "sentiment" in q:
        # try to parse 'emotion: X'
        m = re.search(r"emotion\s*[:\-]\s*([a-z]+)", q)
        if m: emo = _Q_EMO.get(m.group(1), m.group(1))
        qtype = "emotion"
    if emo and qtype == "statement":
        qtype = "emotion"
    return {"type": qtype, "emotion": emo}

# --------------------------
# Build query entry (process -> encoders)
# --------------------------
def build_query(image_path=None, audio_path=None, text_input=None) -> Dict[str, Any]:
    entry: Dict[str, Any] = {}

    if image_path:
        proc = process_image(image_path)
        tensor = load_image(image_path)
        with torch.no_grad():
            key = img_enc(tensor).cpu().numpy().flatten().tolist()
        value = sem_model.encode(proc["caption"]).tolist()
        entry.update({
            "type": "image",
            "query_text": proc["caption"],          # natural language representation
            "key": key,                             # 15D
            "value": value,                         # 384D
            "meta": {
                "emotion": proc.get("emotion", ""),
                "importance": proc.get("importance", 0.0),
                "caption": proc.get("caption", "")
            }
        })

    if audio_path:
        proc = process_audio(audio_path)
        tensor = load_audio(audio_path)
        with torch.no_grad():
            key = aud_enc(tensor).cpu().numpy().flatten().tolist()
        value = sem_model.encode(proc.get("transcribed", "") or "non-speech").tolist()
        entry.update({
            "type": "audio",
            "query_text": proc.get("audio_text", proc.get("transcribed", "audio input")),
            "key": key,
            "value": value,
            "meta": {
                "emotion": proc.get("emotion", ""),
                "importance": proc.get("importance", 0.0),
                "transcribed": proc.get("transcribed", "")
            }
        })

    if text_input:
        proc = process_text(text_input)
        with torch.no_grad():
            key = txt_enc([text_input]).cpu().numpy().flatten().tolist()
        value = sem_model.encode(text_input).tolist()
        entry.update({
            "type": "text",
            "query_text": proc.get("text_summary", text_input),
            "key": key,
            "value": value,
            "meta": {
                "emotion": proc.get("emotion", ""),
                "importance": proc.get("importance", 0.0),
                "raw_text": text_input
            }
        })

    return entry

# --------------------------
# Retrieval (cross-modal) + keyword & modality boosts
# --------------------------
def load_memory(path=MEMORY_FILE) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def retrieve_topn(query_entry, memory, top_n=TOP_N, alpha: float = ALPHA) -> List[Dict[str, Any]]:
    """
    Rank memory entries by a fusion of:
      - semantic similarity (384D value)  -> primary
      - key-space similarity (15D key)    -> secondary
      - keyword overlap with query text   -> tertiary
      - small same-modality boost
    score = (1-alpha)*sim_value + alpha*sim_key + KEYWORD_BOOST*overlap + MODALITY_BOOST[type]
    """
    qtext = query_entry.get("query_text", "")
    qtype = query_entry.get("type", "")
    scores = []
    for item in memory:
        sv = cosine_sim(query_entry["value"], item["value"])
        sk = cosine_sim(query_entry["key"], item["key"])
        overlap = keyword_overlap(qtext, item.get("caption", ""))
        base = (1 - alpha) * sv + alpha * sk
        modal = item.get("type", "")
        bonus = KEYWORD_BOOST * overlap + MODALITY_BOOST.get(modal, 0.0) * (1.0 if modal in qtype else 0.0)
        score = base + bonus
        scores.append({
            "type": modal,
            "caption": item.get("caption", ""),
            "score": float(score),
            "sim_value": float(sv),
            "sim_key": float(sk),
            "overlap": float(overlap),
            "meta": item.get("processor", {})
        })
    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores[:top_n]

# --------------------------
# Evidence Selection and Reranking
# --------------------------
def rerank_with_entailment(query_entry, retrieved: List[Dict[str, Any]], top_k:int=3) -> List[Dict[str, Any]]:
    """
    For the top_k items, compute a quick 'alignment' score:
      align = 0.7 * score + 0.3 * kw_overlap(query, caption)
    Then verify the caption is NOT contradicting the query using NLI(query->caption).
    Items contradicted by the query get a penalty.
    """
    qtext = query_entry.get("query_text","")
    ranked = []
    for r in retrieved[:max(1, top_k)]:
        kw = r.get("overlap", 0.0)
        base = r.get("score", 0.0)
        align = 0.7*base + 0.3*kw
        try:
            ent_q2cap = float(nli_entailment_score(qtext, r.get("caption","")))
        except Exception:
            ent_q2cap = 0.33
        # If query contradicts caption (low entailment), downweight
        align_adj = align if ent_q2cap >= 0.34 else align - 0.10
        r2 = dict(r)
        r2["align"] = align_adj
        r2["entail_q_to_cap"] = ent_q2cap
        ranked.append(r2)
    ranked.sort(key=lambda x: x["align"], reverse=True)
    return ranked

# --------------------------
# Answer Drafting
# --------------------------
_SUBJ_PAT = re.compile(r"\b(?:a|an|the)\s+([a-z]+)", re.I)
def extract_subject(caption: str) -> str:
    m = _SUBJ_PAT.search(caption or "")
    if m: return m.group(0)  # keep article e.g., "a man"
    toks = tokens_doc(caption or "")
    return toks[0] if toks else "the subject"

def draft_final_answer(query_text: str, evidence: Dict[str, Any]) -> Tuple[str, str]:
    """Return (justification, final_answer) built deterministically from the caption + query intent."""
    caption = evidence.get("caption","").strip()
    subj = extract_subject(caption)
    qinfo = classify_query(query_text)
    qtype, emo = qinfo["type"], qinfo.get("emotion")

    if qtype == "who":
        return (f"The caption states '{caption}'. It mentions {subj}.",
                f"{subj.capitalize()} is the person.")
    if qtype == "where":
        # naive location extraction
        loc = None
        for cand in ["park","street","room","kitchen","beach","sea","ocean","garden","office","bench","grass"]:
            if cand in caption.lower(): loc = cand; break
        if loc:
            return (f"The caption includes the setting '{loc}'.",
                    f"It takes place in the {loc}.")
        return (f"The caption states '{caption}'.", "The location is not specified in the evidence.")
    if qtype == "emotion":
        # prefer explicit "Emotion: X" in caption if present
        m = re.search(r"emotion\s*[:\-]\s*([a-z]+)", caption, flags=re.I)
        label = (m.group(1).lower() if m else (emo or "unknown")).strip()
        return (f"The evidence text says '{caption}'.",
                f"The expressed emotion is {label}.")
    if qtype == "what":
        return (f"The caption states '{caption}'.",
                caption[0].upper() + caption[1:] if caption else "No evidence.")
    if qtype == "yesno":
        # Build a cautious answer: we can't assert yes/no without a proposition;
        # return a grounded statement from the caption.
        return (f"The answer should be grounded in evidence: '{caption}'.",
                caption[0].upper() + caption[1:] if caption else "Insufficient evidence.")
    # default (statement / describe)
    return (f"Using the top aligned evidence: '{caption}'.",
            caption[0].upper() + caption[1:] if caption else "No matching evidence.")

# --------------------------
# End-to-end Reasoning
# --------------------------
def reason_over_candidates(query_entry, retrieved, top_k:int=3) -> Dict[str, Any]:
    """
    1) Rerank first top_k with an alignment that includes keyword overlap + NLI sanity.
    2) For each candidate, draft an answer deterministically.
    3) Use NLI to ensure (caption -> final_answer) is entailed.
    4) Pick the highest entailed candidate; if all low, fall back to most aligned caption.
    Returns a dict with trace.
    """
    # Recompute keyword overlap robustly, in case the caller used a different one
    qtext = query_entry.get("query_text","")
    cand = []
    reranked = rerank_with_entailment(query_entry, retrieved, top_k=top_k)
    for r in reranked:
        r = dict(r)
        r["kw_overlap_fixed"] = keyword_overlap(qtext, r.get("caption",""))
        just, ans = draft_final_answer(qtext, r)
        try:
            ent_cap_to_ans = float(nli_entailment_score(r.get("caption",""), ans))
        except Exception:
            ent_cap_to_ans = 0.33
        r["draft_reason"] = just
        r["draft_final"] = ans
        r["entail_cap_to_final"] = ent_cap_to_ans
        # final selection score
        r["select_score"] = 0.6*r["align"] + 0.4*ent_cap_to_ans
        cand.append(r)

    cand.sort(key=lambda x: x["select_score"], reverse=True)
    chosen = cand[0] if cand else (reranked[0] if reranked else {})
    trace = {
        "query_text": qtext,
        "query_type": classify_query(qtext),
        "candidates": [{
            "caption": c.get("caption",""),
            "align": round(float(c.get("align",0.0)), 3),
            "kw_overlap_fixed": round(float(c.get("kw_overlap_fixed",0.0)), 3),
            "entail_q_to_cap": round(float(c.get("entail_q_to_cap",0.0)), 3),
            "entail_cap_to_final": round(float(c.get("entail_cap_to_final",0.0)), 3),
            "draft_final": c.get("draft_final",""),
            "draft_reason": c.get("draft_reason",""),
            "select_score": round(float(c.get("select_score",0.0)), 3)
        } for c in cand],
        "selected": {
            "caption": chosen.get("caption",""),
            "final": chosen.get("draft_final", chosen.get("caption","")),
            "reason": chosen.get("draft_reason",""),
            "confidence": round(float(chosen.get("entail_cap_to_final",0.0)), 3)
        }
    }
    return trace

# --------------------------
# Fuse multimodal embeddings
# --------------------------
def fuse_embeddings(keys: List[np.ndarray], values: List[np.ndarray]) -> Dict[str, Any]:
    """
    Fuse multiple modality embeddings into a single query representation.
    - key (15D): average-pool across modalities
    - value (384D): average-pool across modalities
    """
    fused_key = np.mean(keys, axis=0).tolist()
    fused_value = np.mean(values, axis=0).tolist()
    return {"key": fused_key, "value": fused_value}

def build_multimodal_query(image_path=None, audio_path=None, text_input=None) -> Dict[str, Any]:
    """
    Build a multimodal query entry. If more than one modality is given,
    we fuse their embeddings while keeping the captions.
    """
    subqueries = []
    if image_path:
        subqueries.append(build_query(image_path=image_path))
    if audio_path:
        subqueries.append(build_query(audio_path=audio_path))
    if text_input:
        subqueries.append(build_query(text_input=text_input))

    if not subqueries:
        raise ValueError("No input modalities provided.")

    if len(subqueries) == 1:
        return subqueries[0]

    # Fuse keys and values
    fused = fuse_embeddings(
        keys=[sq["key"] for sq in subqueries],
        values=[sq["value"] for sq in subqueries]
    )

    # Merge metadata
    captions = [sq["query_text"] for sq in subqueries]
    meta_list = [sq["meta"] for sq in subqueries]

    return {
        "type": "+".join([sq["type"] for sq in subqueries]),
        "query_text": " | ".join(captions),
        "key": fused["key"],
        "value": fused["value"],
        "meta": {
            "sources": meta_list
        }
    }
# ... [Previous imports and configuration remain the same until line 530] ...

def extract_context_elements(text: str) -> List[str]:
    """Extracts contextual elements like locations, actions, etc."""
    context = []
    
    # Location context
    locations = ["park", "street", "room", "kitchen", "beach", "sea", "ocean", 
                "garden", "office", "bench", "grass", "water", "indoors", "outdoors"]
    for loc in locations:
        if loc in text.lower():
            context.append(loc)
    
    # Action context
    actions = ["sitting", "standing", "playing", "running", "walking", 
              "lying", "sleeping", "eating", "drinking"]
    for action in actions:
        if action in text.lower():
            context.append(action)
    
    return context if context else ["general context"]

def extract_semantic_concepts(text: str) -> List[str]:
    """Extracts key semantic concepts from text"""
    # Remove stopwords and keep meaningful terms
    stopwords = {"a", "an", "the", "is", "are", "in", "on", "at"}
    words = [w for w in text.lower().split() if w not in stopwords]
    
    # Simple noun extraction (in a real system, use proper NLP)
    nouns = []
    for word in words:
        if len(word) > 3 and not word.endswith(('ing', 'ed')):  # Basic noun heuristic
            nouns.append(word)
    return nouns if nouns else words  # Fall back to all words if no nouns found

def temporal_relevance_score(item: Dict[str, Any]) -> float:
    """Calculates temporal relevance score (placeholder implementation)"""
    # In a real system, this would consider timestamps, sequence, etc.
    return 0.5  # Neutral score since we don't have temporal data

def emotional_alignment(query: Dict[str, Any], item: Dict[str, Any]) -> float:
    """Calculates emotional alignment between query and item"""
    query_emo = query.get("meta", {}).get("emotion", "").lower()
    item_emo = item.get("meta", {}).get("emotion", "").lower()
    
    if not query_emo or not item_emo:
        return 0.5  # Neutral if either lacks emotion
    
    # Simple emotion matching
    positive = {"happy", "joy", "excited"}
    negative = {"sad", "anger", "fear", "disgust"}
    
    if (query_emo in positive and item_emo in positive) or \
       (query_emo in negative and item_emo in negative):
        return 1.0
    elif (query_emo in positive and item_emo in negative) or \
         (query_emo in negative and item_emo in positive):
        return 0.0
    return 0.5  # Neutral for mixed/unknown cases

def check_physical_plausibility(item: Dict[str, Any]) -> float:
    """Checks if the memory is physically plausible"""
    # Placeholder - in a real system this would check physics constraints
    return 1.0  # Assume all memories are plausible

def check_contextual_fit(item: Dict[str, Any], context_elements: List[str]) -> float:
    """Checks how well the item fits the query context"""
    caption = item.get("caption", "").lower()
    matches = sum(1 for ctx in context_elements if ctx in caption)
    return matches / (len(context_elements) + 1e-9)

def calculate_confidence_metrics(query: str, item: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Calculates various confidence metrics"""
    if not item:
        return {}
    
    caption = item.get("caption", "")
    return {
        "visual_confidence": item.get("sim_value", 0),
        "conceptual_confidence": conceptual_alignment_score(query, caption),
        "contextual_confidence": check_contextual_fit(item, extract_context_elements(query)),
        "emotional_confidence": emotional_alignment({"meta": {"emotion": ""}}, item)
    }

def identify_uncertainties(query: str, item: Optional[Dict[str, Any]], 
                         memory_analysis: List[Dict[str, Any]]) -> List[str]:
    """Identifies remaining uncertainties about the answer"""
    uncertainties = []
    
    if not item:
        return ["No matching items found in memory"]
    
    # Check for visual uncertainty
    if memory_analysis and memory_analysis[0]['similarity_breakdown']['visual'] < 0.5:
        uncertainties.append("Low visual similarity to query")
    
    # Check for conceptual gaps
    query_concepts = set(extract_semantic_concepts(query))
    item_concepts = set(extract_semantic_concepts(item.get("caption", "")))
    missing_concepts = query_concepts - item_concepts
    if missing_concepts:
        uncertainties.append(f"Missing concepts: {', '.join(missing_concepts)}")
    
    # Check for reference ambiguity
    if "this" in query.lower() or "that" in query.lower():
        uncertainties.append("Potential ambiguity in reference resolution")
    
    return uncertainties if uncertainties else ["No significant uncertainties"]

# ... [Rest of the code remains the same] ...
# --------------------------
# Updated end-to-end run
# --------------------------
import os
import json
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# ... [all your other existing imports] ...

class QueryType(Enum):
    WHO = "who"
    WHAT = "what"
    WHERE = "where"
    YESNO = "yesno"
    EMOTION = "emotion"
    STATEMENT = "statement"
    COMPARISON = "comparison"
    REFERENCE = "reference"

@dataclass
class CognitiveAnalysis:
    query_understanding: Dict[str, Any]
    memory_analysis: List[Dict[str, Any]]
    comparative_reasoning: List[str]
    conclusion: Dict[str, Any]
    confidence_metrics: Dict[str, float]
    uncertainties: List[str]

def extract_context_elements(text: str) -> List[str]:
    """Extracts contextual elements like locations, actions, etc."""
    context = []
    locations = ["park", "street", "room", "kitchen", "beach", "sea", "ocean", 
                "garden", "office", "bench", "grass", "water", "indoors", "outdoors"]
    actions = ["sitting", "standing", "playing", "running", "walking", 
              "lying", "sleeping", "eating", "drinking"]
    
    text_lower = text.lower()
    context.extend(loc for loc in locations if loc in text_lower)
    context.extend(action for action in actions if action in text_lower)
    
    return context if context else ["general context"]

def extract_semantic_concepts(text: str) -> List[str]:
    """Extracts key semantic concepts from text"""
    stopwords = {"a", "an", "the", "is", "are", "in", "on", "at"}
    words = [w for w in text.lower().split() if w not in stopwords and len(w) > 2]
    return words

def temporal_relevance_score(item: Dict[str, Any]) -> float:
    """Placeholder for temporal relevance scoring"""
    return 0.5  # Neutral default

def emotional_alignment(query: Dict[str, Any], item: Dict[str, Any]) -> float:
    """Calculates emotional alignment between query and item"""
    query_emo = query.get("meta", {}).get("emotion", "").lower()
    item_emo = item.get("meta", {}).get("emotion", "").lower()
    
    if not query_emo or not item_emo:
        return 0.5
    
    positive = {"happy", "joy", "excited"}
    negative = {"sad", "anger", "fear", "disgust"}
    
    if (query_emo in positive and item_emo in positive) or \
       (query_emo in negative and item_emo in negative):
        return 1.0
    return 0.0 if (query_emo in positive) != (item_emo in positive) else 0.5

def check_physical_plausibility(item: Dict[str, Any]) -> float:
    """Placeholder for physical plausibility check"""
    return 1.0  # Assume all are plausible

def check_contextual_fit(item: Dict[str, Any], context_elements: List[str]) -> float:
    """Checks how well item fits query context"""
    caption = item.get("caption", "").lower()
    matches = sum(1 for ctx in context_elements if ctx in caption)
    return matches / max(1, len(context_elements))

def calculate_confidence_metrics(query: str, item: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Calculates confidence metrics for a memory item"""
    if not item:
        return {}
    
    return {
        "visual_confidence": item.get("sim_value", 0),
        "conceptual_confidence": conceptual_alignment_score(query, item.get("caption", "")),
        "contextual_confidence": check_contextual_fit(item, extract_context_elements(query)),
        "emotional_confidence": emotional_alignment({"meta": {}}, item)
    }

def identify_uncertainties(query: str, item: Optional[Dict[str, Any]], 
                         memory_analysis: List[Dict[str, Any]]) -> List[str]:
    """Identifies uncertainties about the answer"""
    if not item:
        return ["No matching items found"]
    
    uncertainties = []
    if memory_analysis and memory_analysis[0]['similarity_breakdown']['visual'] < 0.5:
        uncertainties.append("Low visual similarity")
    
    query_concepts = set(extract_semantic_concepts(query))
    item_concepts = set(extract_semantic_concepts(item.get("caption", "")))
    missing = query_concepts - item_concepts
    if missing:
        uncertainties.append(f"Missing concepts: {', '.join(missing)}")
    
    if "this" in query.lower() or "that" in query.lower():
        uncertainties.append("Potential reference ambiguity")
    
    return uncertainties or ["No significant uncertainties"]

def classify_deep_query(qtext: str) -> QueryType:
    """Enhanced query classification"""
    qn = norm(qtext)
    if "same" in qn or "again" in qn:
        return QueryType.COMPARISON
    if "this" in qn or "that" in qn:
        return QueryType.REFERENCE
    
    # Fall back to original classification
    original = classify_query(qtext)
    return QueryType(original["type"].upper())

def conceptual_alignment_score(query: str, candidate: str) -> float:
    """Calculates conceptual alignment score"""
    q_concepts = set(extract_semantic_concepts(query))
    c_concepts = set(extract_semantic_concepts(candidate))
    return len(q_concepts & c_concepts) / max(1, len(q_concepts))

def generate_comparative_analysis(query: str, items: List[Dict[str, Any]], 
                                analysis: List[Dict[str, Any]]) -> List[str]:
    """Generates comparative analysis between top items"""
    if len(items) < 2:
        return []
    
    insights = [
        f"Top match '{items[0]['caption']}' scores {items[0]['score']:.2f} "
        f"(conceptual: {analysis[0]['similarity_breakdown']['conceptual']:.2f})"
    ]
    
    if analysis[0]['similarity_breakdown']['visual'] > 0.7:
        insights.append("Strong visual match")
    if analysis[1]['plausibility']['contextual_fit'] > 0.7:
        insights.append("Runner-up has better contextual fit")
    
    return insights

def generate_explanation(query: str, item: Optional[Dict[str, Any]]) -> str:
    """Generates explanation for selected item"""
    if not item:
        return "No matching items found"
    
    parts = [f"Memory shows: '{item.get('caption', '')}'"]
    concepts = extract_semantic_concepts(query)
    matched = [c for c in concepts if c in item.get("caption", "").lower()]
    if matched:
        parts.append(f"Matches concepts: {', '.join(matched)}")
    
    if "emotion" in query.lower():
        emo = item.get("meta", {}).get("emotion", "")
        if emo:
            parts.append(f"Emotion: {emo}")
    
    return ". ".join(parts) + "."

def generate_deep_reasoning_output(cog_analysis: CognitiveAnalysis) -> str:
    """Formats cognitive analysis as readable output"""
    output = [
        "=== DEEP REASONING ===",
        f"Query type: {cog_analysis.query_understanding['type'].value}",
        f"Main subject: {cog_analysis.query_understanding['main_subject']}",
        ""
    ]
    
    for i, mem in enumerate(cog_analysis.memory_analysis, 1):
        output.extend([
            f"Item {i}: {mem['item']['caption']}",
            f"- Visual: {mem['similarity_breakdown']['visual']:.2f}",
            f"- Conceptual: {mem['similarity_breakdown']['conceptual']:.2f}",
            ""
        ])
    
    if cog_analysis.comparative_reasoning:
        output.append("Comparative insights:")
        output.extend(f"- {insight}" for insight in cog_analysis.comparative_reasoning)
        output.append("")
    
    output.extend([
        "Conclusion:",
        cog_analysis.conclusion['explanation'],
        "",
        "Confidence:"
    ])
    output.extend(f"- {k}: {v:.2f}" for k, v in cog_analysis.confidence_metrics.items())
    
    if cog_analysis.uncertainties:
        output.extend(["", "Uncertainties:"] + [f"- {u}" for u in cog_analysis.uncertainties])
    
    return "\n".join(output)

def enhanced_cognitive_processing(query_entry: Dict[str, Any], 
                                retrieved: List[Dict[str, Any]]) -> CognitiveAnalysis:
    """Performs deep cognitive analysis of query against memories"""
    qtext = query_entry.get("query_text", "")
    
    query_type = classify_deep_query(qtext)
    main_subject = extract_subject(qtext)
    context = extract_context_elements(qtext)
    
    memory_analysis = []
    for item in retrieved[:3]:
        memory_analysis.append({
            "item": item,
            "similarity_breakdown": {
                "visual": item.get("sim_value", 0),
                "conceptual": conceptual_alignment_score(qtext, item.get("caption", "")),
                "temporal": temporal_relevance_score(item),
                "emotional": emotional_alignment(query_entry, item)
            },
            "plausibility": {
                "physical_constraints": check_physical_plausibility(item),
                "contextual_fit": check_contextual_fit(item, context)
            }
        })
    
    comparative = generate_comparative_analysis(qtext, retrieved[:3], memory_analysis)
    top_item = retrieved[0] if retrieved else None
    confidence = calculate_confidence_metrics(qtext, top_item)
    uncertainties = identify_uncertainties(qtext, top_item, memory_analysis)
    
    return CognitiveAnalysis(
        query_understanding={
            "type": query_type,
            "main_subject": main_subject,
            "context": context,
            "emotional_tone": query_entry.get("meta", {}).get("emotion", "")
        },
        memory_analysis=memory_analysis,
        comparative_reasoning=comparative,
        conclusion={
            "selected_item": top_item,
            "explanation": generate_explanation(qtext, top_item)
        },
        confidence_metrics=confidence,
        uncertainties=uncertainties
    )

# Update the main run function
def run_reasoned_query(image_path=None, audio_path=None, text_input=None, top_n=TOP_N):
    memory = load_memory(MEMORY_FILE)
    if not memory:
        raise RuntimeError(f"No memory found at {MEMORY_FILE}")

    query = build_multimodal_query(image_path, audio_path, text_input)
    results = retrieve_topn(query, memory, top_n, ALPHA)
    
    # Original reasoning
    reasoning_trace = reason_over_candidates(query, results, min(3, top_n))
    selected = reasoning_trace["selected"]
    
    # Enhanced cognitive processing
    cog_analysis = enhanced_cognitive_processing(query, results)
    deep_reasoning = generate_deep_reasoning_output(cog_analysis)
    
    # Generate final output
    prompt = f"""Combine these into a coherent response:
    
    Brief answer: {selected['final']}
    
    Detailed reasoning:
    {deep_reasoning}
    
    Format as:
    Final Answer: <answer>
    Reasoning: <explanation>
    Confidence: <level>"""
    
    final_output = run_reasoner(prompt, max_new_tokens=256)
    
    # Print outputs
    print("\n=== STANDARD OUTPUT ===")
    print(f"Final: {selected['final']}")
    print(f"Reasoning: {selected['reason']}")
    
    print("\n=== DEEP REASONING ===")
    print(deep_reasoning)
    
    print("\n=== FINAL OUTPUT ===")
    print(final_output)
    
    return {
        "query": query,
        "matches": results,
        "reasoning_trace": reasoning_trace,
        "cognitive_analysis": cog_analysis,
        "final_output": final_output
    }

# ... [rest of your existing code remains unchanged] ...
# ... [keep all remaining existing functions] ...

if __name__ == "__main__":
    # Example with deep reasoning
    out = run_reasoned_query(
        image_path="data/download.jpeg",
        text_input="Is this the same dog we saw earlier?",
        top_n=5
    )