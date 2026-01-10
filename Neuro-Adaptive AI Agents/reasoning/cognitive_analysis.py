# reasoning/cognitive_analysis.py
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from reasoning.query_understanding import extract_subject, classify_query
from reasoning.semantic_utils import extract_semantic_concepts
from reasoning.answer_generation import generate_explanation, generate_comparative_analysis

@dataclass
class CognitiveAnalysis:
    query_understanding: Dict[str, Any]
    memory_analysis: List[Dict[str, Any]]
    comparative_reasoning: List[str]
    conclusion: Dict[str, Any]
    confidence_metrics: Dict[str, float]
    uncertainties: List[str]

def extract_context_elements(text: str) -> List[str]:
    context = []
    locations = ["park", "street", "room", "kitchen", "beach", "sea", "ocean", 
                "garden", "office", "bench", "grass", "water", "indoors", "outdoors"]
    actions = ["sitting", "standing", "playing", "running", "walking", 
              "lying", "sleeping", "eating", "drinking"]
    
    text_lower = text.lower()
    context.extend(loc for loc in locations if loc in text_lower)
    context.extend(action for action in actions if action in text_lower)
    
    return context if context else ["general context"]

def conceptual_alignment_score(query: str, candidate: str) -> float:
    q_concepts = set(extract_semantic_concepts(query))
    c_concepts = set(extract_semantic_concepts(candidate))
    return len(q_concepts & c_concepts) / max(1, len(q_concepts))

def emotional_alignment(query: Dict[str, Any], item: Dict[str, Any]) -> float:
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

def check_contextual_fit(item: Dict[str, Any], context_elements: List[str]) -> float:
    caption = item.get("caption", "").lower()
    matches = sum(1 for ctx in context_elements if ctx in caption)
    return matches / max(1, len(context_elements))

def calculate_confidence_metrics(query: str, item: Optional[Dict[str, Any]]) -> Dict[str, float]:
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

def enhanced_cognitive_processing(query_entry: Dict[str, Any], 
                                retrieved: List[Dict[str, Any]],
                                vision_data: Dict[str, Any] = None) -> CognitiveAnalysis:
    """Standalone cognitive processing without self reference"""
    
    # Handle list input if needed
    if isinstance(query_entry, list) and query_entry:
        query_entry = query_entry[0]
    
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
                "object_based": _calculate_object_similarity(vision_data, item) if vision_data else 0.0
            },
            "plausibility": {
                "physical_constraints": 1.0,
                "contextual_fit": check_contextual_fit(item, context)
            }
        })
    
    comparative = generate_comparative_analysis(qtext, retrieved[:3], memory_analysis)
    
    if vision_data:
        comparative.extend(_generate_object_based_insights(vision_data, retrieved))
    
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

# Helper functions
def _calculate_object_similarity(vision_data: Dict[str, Any], memory_item: Dict[str, Any]) -> float:
    if not vision_data or 'objects_detected' not in vision_data:
        return 0.0
    vision_objects = [obj['class'] for obj in vision_data['objects_detected']]
    memory_caption = memory_item.get('caption', '').lower()
    memory_objects = [obj for obj in vision_objects if obj in memory_caption]
    return len(memory_objects) / max(1, len(vision_objects))

def _generate_object_based_insights(vision_data: Dict[str, Any], retrieved: List[Dict[str, Any]]) -> List[str]:
    insights = []
    if vision_data and 'objects_detected' in vision_data:
        vision_objects = [obj['class'] for obj in vision_data['objects_detected']]
        for i, item in enumerate(retrieved[:2]):
            memory_caption = item.get('caption', '').lower()
            matching_objects = [obj for obj in vision_objects if obj in memory_caption]
            if matching_objects:
                insights.append(f"Memory {i+1} shares objects: {', '.join(matching_objects)}")
    return insights
