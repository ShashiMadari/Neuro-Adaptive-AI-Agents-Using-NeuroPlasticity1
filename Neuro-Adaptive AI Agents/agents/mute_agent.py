import numpy as np
from typing import Any, Dict, List
from datetime import datetime

from .base_agent import NeuroAdaptiveAgent
from reasoning.extended_vision import get_vision_reasoner
from core.query_builder import build_query
from reasoning.gemini_flash_reasoner import get_flash_reasoner
from .memory_agent import LessHearableAgent
from sentence_transformers import SentenceTransformer
import torch
from torchvision import transforms
from PIL import Image
import librosa
import os
from .output_layer import OutputLayer
from typing import Dict, Any, Optional

class MutePersonAgent(NeuroAdaptiveAgent):
    """
    Agent specialized for visually-centered reasoning
    tailored for mute-person assistive interpretation.
    Audio is intentionally ignored; emphasis is on images and captions.
    """

    def __init__(self, memory_manager, img_enc, txt_enc, device,manager=None,
                 less_hearable_agent=None, api_key: str = None, output_layer: OutputLayer = None):
        super().__init__(memory_manager)
        self.agent_type = "mute_person"
        self.img_enc = img_enc
        self.txt_enc = txt_enc
        self.device = device
        self.vision_reasoner = get_vision_reasoner()
        self.flash_reasoner = get_flash_reasoner(api_key)
        self.caption_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.summary = None
        self.manager = manager

        # ✅ Memory integration
        if less_hearable_agent is not None:
            self.less_hearable_agent = less_hearable_agent
        else:
            self.less_hearable_agent = LessHearableAgent(
                txt_enc=txt_enc,
                memory_store=[],
                device=device,
                api_key=api_key
            )

        # ✅ OutputLayer integration
        self.output_layer = output_layer
        if output_layer is None:
            print("[Warning] No OutputLayer provided to MutePersonAgent.")

    def load_image(self, path):
        """Helper to preprocess image"""
        img_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])
        return img_tf(Image.open(path).convert("RGB")).unsqueeze(0).to(self.device)



    def process_modalities(
            self,
            image_path: str = None,
            audio_path: str = None,
            text_input: str = None,
            top_n: int = 5
        ) -> Dict[str, Any]:
        """Mute agent ignores audio, focuses on image + text"""
        
        # query is now a List[Dict[str, Any]] containing image and/or text entries
        query = build_query(
            image_path=image_path,
            audio_path=None,  # always ignore audio
            text_input=text_input,
            img_enc=self.img_enc,
            aud_enc=None,
            txt_enc=self.txt_enc,
            device=self.device
        )

        # retriever.retrieve_topn handles the list of queries
        results = self.retriever.retrieve_topn(query, top_n=top_n)
        top_match = results[0] if results else {}

        vision_data = {}
        if image_path:
            try:
                vision_data = self.vision_reasoner.analyze_image_comprehension(image_path, text_input)
            except Exception:
                vision_data = {}

        flash_reasoning = None
        if self.flash_reasoner:
            # NOTE: If enhanced_cognitive_processing is used elsewhere, you might want to call it here
            # to generate the full cognitive context, instead of manually creating the dictionary.
            # However, following the provided structure:
            cognitive_context = {
                "top_match": top_match,
                "vision_data": vision_data,
                "caption": top_match.get("caption", "")
            }
            
            try:
                # generate_advanced_reasoning is passed the list of queries and the context dict
                flash_reasoning = self.flash_reasoner.generate_advanced_reasoning(
                    query,
                    cognitive_context
                )
            except Exception as e:
                flash_reasoning = f"[Gemini error: {e}]"

        # NOTE: Assuming datetime is imported in your actual file
        from datetime import datetime
        
        return {
            "query": query, # The list of query entries
            "all_results": results,
            "top_match": top_match,
            "vision_data": vision_data,
            "flash_reasoning": flash_reasoning,
            "timestamp": datetime.now().isoformat()
        }

    def generate_reasoned_sentence(self, cognitive_data: Dict[str, Any]) -> str:
        """Prefer Gemini flash reasoning, fallback to vision analysis"""
        if "flash_reasoning" in cognitive_data and cognitive_data["flash_reasoning"]:
            return cognitive_data["flash_reasoning"]

        top_match = cognitive_data.get("top_match", {})
        vision_data = cognitive_data.get("vision_data", {})
        caption = top_match.get("caption", "")

        objects = self._extract_objects_from_caption(caption)
        manipulation = self._analyze_manipulation_from_text(caption)

        if vision_data:
            objects.extend([obj["class"] for obj in vision_data.get("objects_detected", [])])
            manipulation["held_objects"].extend(
                vision_data.get("manipulation_analysis", {}).get("held_objects", [])
            )

        reasoned_sentence = "Visual analysis identifies "
        reasoned_sentence += ", ".join(objects[:3]) if objects else "unidentified objects"

        if manipulation["held_objects"]:
            reasoned_sentence += f", with active interaction involving {', '.join(manipulation['held_objects'])}. "
        else:
            reasoned_sentence += " in a static arrangement. "

        sim_value = top_match.get("sim_value", 0)
        reasoned_sentence += f"(Image similarity confidence: {sim_value:.0%})."
        return reasoned_sentence

    def process_and_forward(self, image_path=None, audio_path=None, text_input=None, top_n=5) -> Dict[str, Any]:
        """Process, generate reasoning, store in memory, and query OutputLayer"""
        processed = self.process_modalities(image_path=image_path, text_input=text_input, top_n=top_n)
        caption = self.generate_reasoned_sentence(processed)

        # Encode caption to 384-dim embedding
        caption_embedding = self.caption_encoder.encode(caption, convert_to_numpy=True).flatten().tolist()

        query_key = None
        if image_path:
            tensor = self.load_image(image_path)
            with torch.no_grad():
                img_embedding = self.img_enc(tensor).cpu().numpy().flatten().tolist()
            query_key = img_embedding  # ✅ always use image as key

        # Fallback: use caption embedding if no image
        if query_key is None:
            query_key = caption_embedding

        # Forward to memory and capture summary
        summary = None
        if self.less_hearable_agent:
            record, summary = self.less_hearable_agent.summarize_and_store(
                query_key=query_key,
                reasoned_sentence=caption,
                value_vec=caption_embedding
            )
            self.summary = summary
            print("summary:", summary)

        # ✅ Query OutputLayer with summary (preferred) or caption
        knowledge_output = None
        if self.output_layer:
            query_text = self.summary if self.summary else caption
            try:
                knowledge_output = self.output_layer.run(query_text, top_n=top_n)
            except Exception as e:
                knowledge_output = {"error": f"OutputLayer query failed: {e}"}

        return {
            "reasoned_sentence": caption,
            "caption_embedding": caption_embedding,
            "query_key": query_key,
            "less_hearable_summary": summary,
            "knowledge_result": knowledge_output
        }

    # ----------------------------- helpers
    def _extract_objects_from_caption(self, caption: str) -> list:
        import re
        words = re.findall(r'\b[a-zA-Z]{3,}\b', caption)
        stop_words = {'the', 'and', 'with', 'while', 'is', 'are', 'this', 'that', 'there'}
        return [word for word in words if word.lower() not in stop_words][:5]

    def _analyze_manipulation_from_text(self, caption: str) -> dict:
        manipulation_verbs = {'holding', 'using', 'manipulating', 'touching', 'grasping', 'operating'}
        words = caption.lower().split()
        held_objects = []
        for i, word in enumerate(words):
            if word in manipulation_verbs and i + 1 < len(words):
                next_word = words[i + 1]
                if len(next_word) > 2:
                    held_objects.append(next_word)
        return {
            'held_objects': held_objects,
            'manipulation_verbs': [v for v in words if v in manipulation_verbs]
        }
