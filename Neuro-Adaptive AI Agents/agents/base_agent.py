from abc import ABC, abstractmethod
import re
from typing import Dict, Any, List
from memory.retrieval import MemoryRetriever
from reasoning.cognitive_analysis import CognitiveAnalysis, extract_context_elements

from datetime import datetime
from reasoning.query_understanding import classify_query, extract_subject 
class NeuroAdaptiveAgent(ABC):
    def __init__(self, memory_manager):
        self.retriever = MemoryRetriever(memory_manager)
        self.agent_type = "base"
        
    
    @abstractmethod
    def process_modalities(self, image_path=None, audio_path=None, text_input=None, top_n=5) -> Dict[str, Any]:
        """Process inputs based on agent's specialization"""
        pass
    
    @abstractmethod
    def generate_reasoned_sentence(self, cognitive_data: Dict[str, Any], vision_data: Dict[str, Any] = None) -> str:
        """Generate agent-specific reasoned output"""
        pass

    def _extract_objects_from_caption(self, caption: str) -> List[str]:
        """Extract objects from caption text"""
        import re
        # Common object patterns
        object_patterns = [
            r'\b(dog|cat|person|man|woman|child|car|tree|house|book|phone)\b',
            r'\b(ball|cup|glass|bottle|leash|toy|frisbee|laptop)\b'
        ]
        
        objects = []
        for pattern in object_patterns:
            objects.extend(re.findall(pattern, caption.lower()))
        
        return list(set(objects))
    
    def _analyze_manipulation_from_text(self, caption: str) -> Dict[str, Any]:
        """Infer manipulation from text description"""
        manipulation = {
            "held_objects": [],
            "active_actions": [],
            "scene_context": ""
        }
        
        caption_lower = caption.lower()
        
        # Detect holding actions
        holding_verbs = ["holding", "carrying", "grasping", "has", "with"]
        for verb in holding_verbs:
            if verb in caption_lower:
                # Extract object after holding verb
                pattern = f"{verb} (?:a|an|the)? ([a-z]+)"
                matches = re.findall(pattern, caption_lower)
                manipulation["held_objects"].extend(matches)
        
        # Detect actions
        action_verbs = ["playing", "walking", "running", "eating", "drinking", "reading"]
        for verb in action_verbs:
            if verb in caption_lower:
                manipulation["active_actions"].append(verb + "ing")
        
        return manipulation