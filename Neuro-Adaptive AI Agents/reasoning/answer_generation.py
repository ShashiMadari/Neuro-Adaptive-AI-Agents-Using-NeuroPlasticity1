import re
from typing import Tuple, Dict, Any,Optional, List
from reasoning.query_understanding import extract_subject
from reasoning.query_understanding import classify_query
from reasoning.semantic_utils import extract_semantic_concepts
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
    
    return (f"Using the top aligned evidence: '{caption}'.",
            caption[0].upper() + caption[1:] if caption else "No matching evidence.")

def generate_explanation(query: str, item: Optional[Dict[str, Any]]) -> str:
    """Generates explanation for selected item"""
    if not item:
        return "No matching items found"
    
    parts = [f"Memory shows: '{item.get('caption', '')}'"]
    concepts = extract_semantic_concepts(query)
    matched = [c for c in concepts if c in item.get("caption", "").lower()]
    if matched:
        parts.append(f"Matches concepts: {', '.join(matched)}")
    
    # Extract emotion from the item's metadata, not from the query
    emo = item.get("meta", {}).get("emotion", "")
    if emo:
        parts.append(f"Emotion: {emo}")
    elif "emotion" in query.lower():
        # If query is about emotion but item doesn't have one, say it's unknown
        parts.append("Emotion: unknown")
    
    return ". ".join(parts) + "."
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

def generate_deep_reasoning_output(cog_analysis) -> str:
    """Formats cognitive analysis as readable output"""
    # Safely get query type
    query_understanding = cog_analysis.query_understanding
    query_type = query_understanding.get('type', 'unknown')
    
    # Safely get main subject (extract from query if not available)
    main_subject = query_understanding.get('main_subject')
    if not main_subject and hasattr(cog_analysis, 'query_text'):
        # Try to extract subject from query text if available
        main_subject = extract_subject(getattr(cog_analysis, 'query_text', ''))
    
    output = [
        "=== DEEP REASONING ===",
        f"Query type: {query_type}",
        f"Main subject: {main_subject or 'unknown'}",
        ""
    ]
    
    for i, mem in enumerate(cog_analysis.memory_analysis, 1):
        output.extend([
            f"Item {i}: {mem['item'].get('caption', 'no caption')}",
            f"- Visual: {mem['similarity_breakdown'].get('visual', 0):.2f}",
            f"- Conceptual: {mem['similarity_breakdown'].get('conceptual', 0):.2f}",
            ""
        ])
    
    if cog_analysis.comparative_reasoning:
        output.append("Comparative insights:")
        output.extend(f"- {insight}" for insight in cog_analysis.comparative_reasoning)
        output.append("")
    
    output.extend([
        "Conclusion:",
        cog_analysis.conclusion.get('explanation', 'No conclusion'),
        "",
        "Confidence:"
    ])
    
    # Safely get confidence metrics
    confidence_metrics = getattr(cog_analysis, 'confidence_metrics', {})
    output.extend(f"- {k}: {v:.2f}" for k, v in confidence_metrics.items())
    
    if getattr(cog_analysis, 'uncertainties', None):
        output.extend(["", "Uncertainties:"] + [f"- {u}" for u in cog_analysis.uncertainties])
    
    return "\n".join(output)