import os
import re  
import json
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# Import from our organized modules
from config.settings import DEVICE, EMBED_DIM, MEMORY_FILE, TOP_N
from models.encoders import ImageEncoder, AudioEncoder, TextEncoder
from models.reasoner import Reasoner
from models.nli_model import NLIModel
from memory.memory_manager import MemoryManager
from memory.retrieval import MemoryRetriever
from reasoning.query_understanding import classify_query, extract_subject
from reasoning.cognitive_analysis import (
    CognitiveAnalysis, extract_context_elements, extract_semantic_concepts,
    conceptual_alignment_score, emotional_alignment, check_contextual_fit,
    calculate_confidence_metrics, identify_uncertainties
)
from reasoning.answer_generation import (
    draft_final_answer, generate_explanation, generate_comparative_analysis,
    generate_deep_reasoning_output
)
from utils.preprocessing import keyword_overlap
from utils.similarity import cosine_sim
from utils.helpers import load_image, load_audio
from multimodal.fusion import build_multimodal_query
from core.query_builder import build_query
from multimodal.fusion import build_multimodal_query
# Import processors (these remain as separate files)
from processors.image_processor import process_image
from processors.audio_processor import process_audio
from processors.text_processor import process_text
import time
# Add to imports
from vision.object_analyzer import get_object_analyzer
from reasoning.extended_vision import get_vision_reasoner
# Add to imports
import os
from dotenv import load_dotenv
from reasoning.gemini_flash_reasoner import get_flash_reasoner
from agents.agent_manager import AgentManager
# Load environment variables
load_dotenv()

# Initialize Gemini Flash (after model loading)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("⚠️  GEMINI_API_KEY not found in .env file")
    gemini_flash = None
else:
    gemini_flash = get_flash_reasoner(GEMINI_API_KEY)
    print("✅ Gemini 1.5 Flash Reasoner initialized")
print("Starting model loading...")

start_time = time.time()

# --------------------------
# Initialize models
# --------------------------
img_enc = ImageEncoder(out_dim=EMBED_DIM).to(DEVICE)
aud_enc = AudioEncoder(out_dim=EMBED_DIM).to(DEVICE)
txt_enc = TextEncoder(out_dim=EMBED_DIM).to(DEVICE)

# Load trained weights
img_enc.load_state_dict(torch.load("models/img_encoder_15d.pt", map_location=DEVICE))
aud_enc.load_state_dict(torch.load("models/aud_encoder_15d.pt", map_location=DEVICE))
txt_enc.load_state_dict(torch.load("models/txt_encoder_15d.pt", map_location=DEVICE))

img_enc.eval(); 
aud_enc.eval(); 
txt_enc.eval()


# Initialize other models
reasoner = Reasoner()
nli_model = NLIModel()
memory_manager = MemoryManager(MEMORY_FILE)
retriever = MemoryRetriever(memory_manager)
agent_manager = AgentManager(memory_manager, img_enc, aud_enc, txt_enc, DEVICE)
# --------------------------
# Build query function
# --------------------------

# --------------------------
# Enhanced cognitive processing
# --------------------------
def enhanced_cognitive_processing(query_entry: Dict[str, Any], 
                                retrieved: List[Dict[str, Any]],
                                vision_data: Dict[str, Any] = None) -> CognitiveAnalysis:
    qtext = query_entry.get("query_text", "")
    
    query_info = classify_query(qtext)
    main_subject = extract_subject(qtext)
    context = extract_context_elements(qtext)
    
    memory_analysis = []
    for item in retrieved[:3]:
        memory_analysis.append({
            "item": item,
            "similarity_breakdown": {
                "visual": item.get("sim_value", 0),
                "conceptual": conceptual_alignment_score(qtext, item.get("caption", "")),
                "temporal": 0.5,
                "emotional": emotional_alignment(query_entry, item),
                "object_based": _calculate_object_similarity(vision_data, item) if vision_data else 0.0  # REMOVED self.
            },
            "plausibility": {
                "physical_constraints": 1.0,
                "contextual_fit": check_contextual_fit(item, context)
            }
        })
    
    comparative = generate_comparative_analysis(qtext, retrieved[:3], memory_analysis)
    
    if vision_data:
        comparative.extend(_generate_object_based_insights(vision_data, retrieved))  # REMOVED self.
    
    top_item = retrieved[0] if retrieved else None
    confidence = calculate_confidence_metrics(qtext, top_item)
    uncertainties = identify_uncertainties(qtext, top_item, memory_analysis)
    
    # Special handling for comparison queries
    if query_info.get("type") == "comparison" and top_item:
        visual_similarity = top_item.get("sim_value", 0)
        if visual_similarity > 0.8:
            comparative.append("Strong visual match (similarity > 80%) - likely the same subject")
        elif visual_similarity > 0.6:
            comparative.append("Moderate visual match (similarity > 60%) - possibly the same subject")
        else:
            comparative.append("Weak visual match - unlikely to be the same subject")
    
    return CognitiveAnalysis(
        query_understanding=query_info,
        memory_analysis=memory_analysis,
        comparative_reasoning=comparative,
        conclusion={
            "selected_item": top_item,
            "explanation": generate_explanation(qtext, top_item)
        },
        confidence_metrics=confidence,
        uncertainties=uncertainties
    )

    
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
def _calculate_object_similarity(vision_data: Dict[str, Any], memory_item: Dict[str, Any]) -> float:
    """Calculate similarity based on object detection"""
    if not vision_data or 'objects_detected' not in vision_data:
        return 0.0
    
    vision_objects = [obj['class'] for obj in vision_data['objects_detected']]
    memory_caption = memory_item.get('caption', '').lower()
    memory_objects = [obj for obj in vision_objects if obj in memory_caption]
    
    if not vision_objects:
        return 0.0
    
    return len(memory_objects) / len(vision_objects)

def _generate_object_based_insights(vision_data: Dict[str, Any], retrieved: List[Dict[str, Any]]) -> List[str]:
    """Generate insights based on object detection"""
    insights = []
    
    if not vision_data or 'objects_detected' not in vision_data:
        return insights
    
    vision_objects = [obj['class'] for obj in vision_data['objects_detected']]
    
    for i, item in enumerate(retrieved[:2]):
        memory_caption = item.get('caption', '').lower()
        matching_objects = [obj for obj in vision_objects if obj in memory_caption]
        
        if matching_objects:
            insights.append(f"Memory {i+1} shares objects: {', '.join(matching_objects)}")
    
    return insights
def reason_over_candidates(query_entry, retrieved, top_k=3):
    """
    Rerank with entailment and draft answers
    """
    from models.nli_model import NLIModel
    nli = NLIModel()
    
    qtext = query_entry.get("query_text","")
    cand = []
    
    # Handle empty retrieved results
    if not retrieved:
        return {
            "query_text": qtext,
            "query_type": classify_query(qtext),
            "candidates": [],
            "selected": {
                "caption": "",
                "final": "No matching memories found",
                "reason": "No relevant memories retrieved from database",
                "confidence": 0.0
            }
        }
    
    for r in retrieved[:max(1, top_k)]:
        kw = r.get("overlap", 0.0)
        base = r.get("score", 0.0)
        align = 0.7*base + 0.3*kw
        try:
            ent_q2cap = float(nli.entailment_score(qtext, r.get("caption","")))
        except Exception:
            ent_q2cap = 0.33
        
        align_adj = align if ent_q2cap >= 0.34 else align - 0.10
        just, ans = draft_final_answer(qtext, r)
        
        try:
            ent_cap_to_ans = float(nli.entailment_score(r.get("caption",""), ans))
        except Exception:
            ent_cap_to_ans = 0.33
        
        cand.append({
            **r,
            "align": align_adj,
            "entail_q_to_cap": ent_q2cap,
            "draft_reason": just,
            "draft_final": ans,
            "entail_cap_to_final": ent_cap_to_ans,
            "select_score": 0.6*align_adj + 0.4*ent_cap_to_ans
        })
    
    cand.sort(key=lambda x: x["select_score"], reverse=True)
    chosen = cand[0] if cand else (retrieved[0] if retrieved else {})
    
    return {
        "query_text": qtext,
        "query_type": classify_query(qtext),
        "candidates": cand,
        "selected": {
            "caption": chosen.get("caption",""),
            "final": chosen.get("draft_final", chosen.get("caption","")),
            "reason": chosen.get("draft_reason",""),
            "confidence": round(float(chosen.get("entail_cap_to_final",0.0)), 3)
        }
    }
def draft_final_answer(query_text: str, evidence: Dict[str, Any]) -> Tuple[str, str]:
    """Return (justification, final_answer) built deterministically from the caption + query intent."""
    caption = evidence.get("caption", "").strip()
    subj = extract_subject(caption)
    qinfo = classify_query(query_text)
    qtype, emo = qinfo.get("type", "factual"), qinfo.get("emotion", "neutral")

    # Handle empty evidence
    if not caption:
        return ("No evidence available in memory", "Insufficient evidence to answer")

    # Special handling for comparison queries
    if qtype == "comparison":
        visual_similarity = evidence.get("sim_value", 0)
        if visual_similarity > 0.8:
            return (f"Strong visual match (similarity: {visual_similarity:.2f}) with '{caption}'.",
                    f"Yes, this appears to be the same {subj}.")
        elif visual_similarity > 0.6:
            return (f"Moderate visual match (similarity: {visual_similarity:.2f}) with '{caption}'.",
                    f"It might be the same {subj}, but cannot be certain.")
        else:
            return (f"Weak visual match (similarity: {visual_similarity:.2f}) with '{caption}'.",
                    f"No, this does not appear to be the same {subj}.")

    # Handle other query types
    if qtype == "who":
        return (f"The caption states '{caption}'. It mentions {subj}.",
                f"{subj.capitalize()} is the person.")
    if qtype == "where":
        loc = None
        for cand in ["park","street","room","kitchen","beach","sea","ocean","garden","office","bench","grass"]:
            if cand in caption.lower(): loc = cand; break
        if loc:
            return (f"The caption includes the setting '{loc}'.",
                    f"It takes place in the {loc}.")
        return (f"The caption states '{caption}'.", "The location is not specified in the evidence.")
    if qtype == "emotion":
        m = re.search(r"emotion\s*[:\-]\s*([a-z]+)", caption, flags=re.I)
        label = (m.group(1).lower() if m else (emo or "unknown")).strip()
        return (f"The evidence text says '{caption}'.",
                f"The expressed emotion is {label}.")
    if qtype == "what":
        return (f"The caption states '{caption}'.",
                caption[0].upper() + caption[1:] if caption else "No evidence.")
    if qtype == "yesno":
        return (f"The answer should be grounded in evidence: '{caption}'.",
                caption[0].upper() + caption[1:] if caption else "Insufficient evidence.")
    
    # Default fallback for any other query type
    return (f"Based on the memory: '{caption}'",
            f"This appears to show {subj or 'something'} in this context.")
# --------------------------
# Main run function
# --------------------------
def run_reasoned_query(image_path=None, audio_path=None, text_input=None, top_n=TOP_N):
    # Build query
    query = build_query(
        image_path=image_path,
        audio_path=audio_path, 
        text_input=text_input,
        img_enc=img_enc,
        aud_enc=aud_enc,
        txt_enc=txt_enc,
        device=DEVICE
    )
    
    # EXTENDED VISION ANALYSIS - If image provided
    vision_analysis = {}
    if image_path:
        try:
            vision_reasoner = get_vision_reasoner()
            vision_analysis = vision_reasoner.analyze_image_comprehension(
                image_path, 
                query.get('query_text', '')
            )
            print("\n🔍 EXTENDED VISION ANALYSIS:")
            print(f"Objects detected: {[obj['class'] for obj in vision_analysis.get('objects_detected', [])]}")
            print(f"Manipulation: {vision_analysis.get('manipulation_analysis', {})}")
            print(f"Scene context: {vision_analysis.get('scene_context', '')}")
            print(f"Inferred actions: {vision_analysis.get('action_inference', '')}")
            
        except Exception as e:
            print(f"⚠️ Vision analysis failed: {e}")
            vision_analysis = {}
    
    # Retrieve from memory
    results = retriever.retrieve_topn(query, top_n=top_n)
    
    # ✅ ADD THIS MISSING LINE: Reason over candidates
    reasoning_result = reason_over_candidates(query, results, top_k=3)
    
    # ✅ ADD THIS: Get the selected item from reasoning result
    selected = reasoning_result.get("selected", {})
    
    # Enhanced cognitive processing with vision data
    cog_analysis = enhanced_cognitive_processing(query, results, vision_analysis)
    deep_reasoning = generate_deep_reasoning_output(cog_analysis)
    
    # Generate Gemini-powered reasoning if available
    gemini_analysis = ""
    if gemini_flash:
        try:
            # Convert CognitiveAnalysis to dict for Gemini
            analysis_dict = {
                "query_understanding": cog_analysis.query_understanding,
                "memory_analysis": cog_analysis.memory_analysis,
                "comparative_reasoning": cog_analysis.comparative_reasoning,
                "confidence_metrics": cog_analysis.confidence_metrics,
                "uncertainties": cog_analysis.uncertainties,
                "conclusion": cog_analysis.conclusion
            }
            
            gemini_analysis = gemini_flash.generate_advanced_reasoning(
                query.get('query_text', ''), 
                analysis_dict
            )
        except Exception as e:
            gemini_analysis = f"❌ Gemini reasoning failed: {str(e)}"
    
    # Generate final output with error handling
    visual_similarity = selected.get("sim_value", 0) if selected else 0
    visual_match_info = f"Visual similarity: {visual_similarity:.2f}"
    
    if visual_similarity > 0.8:
        visual_match_info += " (strong match)"
    elif visual_similarity > 0.6:
        visual_match_info += " (moderate match)"
    else:
        visual_match_info += " (weak match)"
    
    prompt = f"""You are an AI assistant that provides reasoned answers based on evidence.

EVIDENCE:
{selected.get('final', 'No evidence')}
{visual_match_info}

DEEP REASONING ANALYSIS:
{deep_reasoning}

INSTRUCTIONS:
1. Provide a clear final answer
2. Include brief reasoning explaining the visual similarity
3. State your confidence level (high/medium/low)

RESPONSE FORMAT:
Final Answer: [your clear answer]
Reasoning: [your explanation including visual similarity assessment]
Confidence: [high/medium/low]"""

    final_output = reasoner.generate(prompt, max_new_tokens=256)
    
    # Print enhanced outputs
    print("\n=== STANDARD OUTPUT ===")
    print(f"Final: {selected.get('final', 'No answer')}")
    print(f"Reasoning: {selected.get('reason', 'No reasoning')}")
    
    print("\n=== DEEP REASONING ===")
    print(deep_reasoning)
    
    # Show Gemini analysis if available
    if gemini_analysis:
        print("\n" + "🔍"*20)
        print("GEMINI 1.5 FLASH REASONING")
        print("🔍"*20)
        print(gemini_analysis)
    
    print("\n=== FINAL OUTPUT ===")
    print(final_output)
    
    return {
        "query": query,
        "matches": results,
        "reasoning_result": reasoning_result,  # ✅ Now this exists
        "cognitive_analysis": cog_analysis,
        "gemini_reasoning": gemini_analysis,
        "final_output": final_output
    }
# Add this temporary test to main.py before the main run
def test_modalities():
    """Test each modality separately to find which one is failing"""
    print("Testing image modality...")
    try:
        img_test = build_query(image_path="data/download.jpeg", img_enc=img_enc, device=DEVICE)
        print(f"Image test success: {list(img_test.keys())}")
    except Exception as e:
        print(f"Image test failed: {e}")
    
    print("Testing text modality...")
    try:
        txt_test = build_query(text_input="test query", txt_enc=txt_enc, device=DEVICE)
        print(f"Text test success: {list(txt_test.keys())}")
    except Exception as e:
        print(f"Text test failed: {e}")
    
    print("Testing audio modality...")
    try:
        # You might not have an audio file, so skip if not available
        aud_test = build_query(audio_path="test_audio.wav" if os.path.exists("test_audio.wav") else None, 
                              aud_enc=aud_enc, device=DEVICE)
        print(f"Audio test success: {list(aud_test.keys()) if aud_test else 'No audio file'}")
    except Exception as e:
        print(f"Audio test failed: {e}")

# Call this before the main function
test_modalities()
if __name__ == "__main__":
    # Example usage
    out = run_reasoned_query(
        image_path="data/download.jpeg",
        text_input="What is dog doing in the park?",
        top_n=5
    )