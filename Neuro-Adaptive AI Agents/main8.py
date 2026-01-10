import os
import re  
import json
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
from dotenv import load_dotenv

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
from reasoning.cognitive_analysis import enhanced_cognitive_processing

from reasoning.answer_generation import (
    draft_final_answer, generate_explanation, generate_comparative_analysis,
    generate_deep_reasoning_output
)
from utils.similarity import cosine_sim
from core.query_builder import build_query

# Import agents
from agents.agent_manager import AgentManager

# Import vision and Gemini components
from vision.object_analyzer import get_object_analyzer
from reasoning.extended_vision import get_vision_reasoner
from reasoning.gemini_flash_reasoner import get_flash_reasoner

# Load environment variables
load_dotenv()

# Initialize Gemini Flash
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
gemini_flash = None
if GEMINI_API_KEY:
    try:
        gemini_flash = get_flash_reasoner(GEMINI_API_KEY)
        print("✅ Gemini 1.5 Flash Reasoner initialized")
    except Exception as e:
        print(f"❌ Gemini initialization failed: {e}")
        gemini_flash = None
else:
    print("⚠️  GEMINI_API_KEY not found in .env file")

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

img_enc.eval() 
aud_enc.eval() 
txt_enc.eval()

# Initialize other models
reasoner = Reasoner()
nli_model = NLIModel()
memory_manager = MemoryManager(MEMORY_FILE)
retriever = MemoryRetriever(memory_manager)

# Initialize Agent Manager
agent_manager = AgentManager(memory_manager, img_enc, aud_enc, txt_enc, DEVICE)
print("✅ Multi-Agent System Initialized")

import json

def run_multi_agent_query(agent_type="normal", image_path=None, audio_path=None, text_input=None, top_n=5):
    """
    Run query with specific agent type and return structured results for Flask frontend.
    """
    logs = []  # collect all console messages for frontend
    results_summary = {}

    logs.append(f"🤖 {agent_type.upper()} AGENT ACTIVATED")
    logs.append("=" * 50)
    if agent_type == "blind" and not audio_path:
        msg = "⚠️ No audio input provided. Blind agent skipped reasoning."
        logs.append(msg)
        print(msg)
        return {
            "agent_type": agent_type,
            "reasoned_sentence": "",
            "facts": [],
            "knowledge": [],
            "explanation": {"text": msg},
            "logs": logs
        }

    if agent_type == "mute" and not image_path:
        msg = "⚠️ No image input provided. Mute agent skipped reasoning."
        logs.append(msg)
        print(msg)
        return {
            "agent_type": agent_type,
            "reasoned_sentence": "",
            "facts": [],
            "knowledge": [],
            "explanation": {"text": msg},
            "logs": logs
        }
    # Run agent analysis
    results = agent_manager.run_agent_analysis(
        agent_type=agent_type,
        image_path=image_path,
        audio_path=audio_path,
        text_input=text_input,
        top_n=top_n
    )

    if agent_type == "mute":
        if isinstance(results, tuple):
            results = results[0] if results and isinstance(results[0], dict) else {}
        elif not isinstance(results, dict):
            results = {}

        knowledge_result = results.get("knowledge_result", {})
        if isinstance(knowledge_result, tuple):
            knowledge_result = knowledge_result[0] if knowledge_result and isinstance(knowledge_result[0], dict) else {}
        elif isinstance(knowledge_result, dict):
            knowledge_result = knowledge_result.get("knowledge_result", {})
        else:
            knowledge_result = {}
    else:
        if not isinstance(results, dict):
            results = {}
        knowledge_result = results.get("knowledge_result", {}).get("knowledge_result", {})

    agent_type_res = results.get("agent_type", agent_type)
    reasoned_sentence = results.get("reasoned_sentence", "")
    timestamp = results.get("timestamp", "")

    logs.append(f"AGENT TYPE: {agent_type_res}")
    logs.append(f"REASONED SENTENCE: {reasoned_sentence}")
    logs.append(f"TIMESTAMP: {timestamp}")
    logs.append(f"DEBUG results type: {type(results)}")

    # Extract facts and knowledge
    facts_list, knowledge_list = [], []
    if knowledge_result:
        facts = knowledge_result.get("facts", [])
        for f in facts:
            facts_list.append(f if isinstance(f, dict) else {"fact": str(f)})

        knowledge = knowledge_result.get("knowledge", [])
        for k in knowledge:
            knowledge_list.append(k if isinstance(k, dict) else {"knowledge": str(k)})

        logs.append("KNOWLEDGE OUTPUT:")
        logs.append(json.dumps(knowledge_result, indent=4))

    logs.append(f"✅ Extracted {len(facts_list)} facts and {len(knowledge_list)} knowledge items")

    query_text = results.get("query") or reasoned_sentence or ""

    # ----------- Agent specific responses -----------
    explanation = {}
    if agent_type == "normal":
        explanation = gemini_flash.generate_human_like_explanation(
            query=query_text,
            facts=facts_list,
            knowledge=knowledge_list,
            agent_type="normal"
        )
        logs.append("🧑‍🏫 HUMAN-LIKE EXPLANATION:")
        if isinstance(explanation, dict):
            logs.append(f"📝 Text: {explanation.get('text', 'No text')}")
            if "audio" in explanation:
                logs.append(f"🔊 Audio file saved: {explanation['audio']}")

    elif agent_type == "blind":
        explanation = gemini_flash.generate_blind_audio_explanation(
            query=query_text,
            facts=facts_list,
            knowledge=knowledge_list
        )
        logs.append("🎧 BLIND AUDIO EXPLANATION:")
        if isinstance(explanation, dict) and "audio_file" in explanation:
            logs.append(f"🔊 Blind audio explanation saved: {explanation['audio_file']}")
        else:
            logs.append("⚠️ Audio file was not generated.")

        blind_audio_text = explanation.get("audio_text", "")
        mute_feedback = gemini_flash.generate_agent_feedback(
            input_text=blind_audio_text,
            agent_from="blind",
            agent_to="mute",
            context={"facts": facts_list, "knowledge": knowledge_list}
        )
        logs.append("✍️ MUTE FEEDBACK GENERATED FROM BLIND AGENT:")
        logs.append(mute_feedback.get("text", "⚠️ No mute feedback generated."))

    elif agent_type == "mute":
        explanation = gemini_flash.generate_mute_text_explanation(
            query=query_text,
            facts=facts_list,
            knowledge=knowledge_list
        )
        logs.append("✍️ MUTE TEXT EXPLANATION:")
        text_exp = explanation.get("text", "⚠️ No mute explanation generated.")
        logs.append(text_exp)

        blind_feedback = gemini_flash.generate_agent_feedback(
            input_text=text_exp,
            agent_from="mute",
            agent_to="blind",
            context={"facts": facts_list, "knowledge": knowledge_list}
        )
        logs.append("🎯 BLIND FEEDBACK GENERATED FROM MUTE AGENT:")
        if isinstance(blind_feedback, dict) and "audio_file" in blind_feedback:
            logs.append(f"🔊 Blind agent audio feedback saved: {blind_feedback['audio_file']}")
        else:
            logs.append("⚠️ Blind feedback audio was not generated.")

    # ----------- Final packaged response -----------
    results_summary = {
        "agent_type": agent_type_res,
        "reasoned_sentence": reasoned_sentence,
        "timestamp": timestamp,
        "facts": facts_list,
        "knowledge": knowledge_list,
        "explanation": explanation,
        "logs": logs
    }

    return results_summary




    
    # Show additional details based on agent type
    '''if agent_type == "normal" and 'cognitive_analysis' in results['results']:
        print("\n📊 COGNITIVE ANALYSIS:")
        analysis = results['results']['cognitive_analysis']
        if hasattr(analysis, 'confidence_metrics'):
            for metric, value in analysis.confidence_metrics.items():
                print(f"  {metric}: {value:.2f}")
    
    elif agent_type == "memory" and 'memory_stats' in results['results']:
        print("\n💾 MEMORY STATISTICS:")
        stats = results['results']['memory_stats']
        print(f"  Total memories: {stats.get('total_memories', 0)}")
        print(f"  Domain relevant: {stats.get('domain_relevant', 0)}")
    '''
    

# --------------------------
# Original run function (for backward compatibility)
# --------------------------
def run_reasoned_query(image_path=None, audio_path=None, text_input=None, top_n=TOP_N):
    """Original function that uses normal agent"""
    return run_multi_agent_query(
        agent_type="normal",
        image_path=image_path,
        audio_path=audio_path,
        text_input=text_input,
        top_n=top_n
    )

# --------------------------
# Test modalities
# --------------------------
def test_modalities():
    """Test each modality separately"""
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
        aud_test = build_query(audio_path="einstein_formula.wav", 
                              aud_enc=aud_enc, device=DEVICE)
        print(f"Audio test success: {list(aud_test.keys()) if aud_test else 'No audio file'}")
    except Exception as e:
        print(f"Audio test failed: {e}")

# --------------------------
# Main execution
# --------------------------

if __name__ == "__main__":
    # Test modalities first
    test_modalities()
    
    print("\n" + "="*60)
    print("NEURO-ADAPTIVE MULTI-AGENT SYSTEM DEMO")
    print("="*60)
    
    # Demo all agents
    agents_to_test = ["normal"]  
    
    for agent_type in agents_to_test:
        try:
            if agent_type == "blind":
                # Blind agent - audio focused
                result = run_multi_agent_query(
                    agent_type=agent_type,
                    audio_path="einstein_formula.wav",
                    text_input="Explain this formula?",
                    top_n=5
                )
            elif agent_type == "mute":
                # Mute agent - image + text only
                result = run_multi_agent_query(
                agent_type=agent_type,
                image_path="data/mc2.png",
                text_input="Explain please einstein_formula?",
                top_n=5
            )
            
            else:
                # Normal agent - all modalities
                result = run_multi_agent_query(
                    agent_type=agent_type,
                    image_path="data/mc2.png",
                    audio_path="einstein_formula.wav",
                    text_input="Explain about this formula?",
                    top_n=5
                )
                # ✅ After normal agent: show memory contents
                '''print("\n🧠 MEMORY CONTENTS AFTER NORMAL AGENT:")
                memory_data = memory_manager.load_memory()
                for k, v in memory_data.items():
                    print(f"  [{k}] → Modality: {v.get('type', 'unknown')}")
                    print(f"       Summary: {v.get('caption', '---')}")
                    if 'path' in v:
                        print(f"       Path: {v['path']}")
                    if 'processor' in v:
                        print(f"       Processor Keys: {list(v['processor'].keys())}")
                    print("-" * 40)
            
            print(f"\n✅ {agent_type.upper()} AGENT COMPLETED SUCCESSFULLY")
            print("-" * 50)
            '''
        except Exception as e:
            print(f"❌ {agent_type.upper()} AGENT FAILED: {e}")
            continue
    
    print("\n🎯 MULTI-AGENT SYSTEM DEMO COMPLETED!")
    print(f"Total agents tested: {len(agents_to_test)}")