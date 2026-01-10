# ----------------------------------
# normal_person_agent.py
# ----------------------------------
import numpy as np
from typing import Any, Dict
from .base_agent import NeuroAdaptiveAgent
from reasoning.extended_vision import get_vision_reasoner
from core.query_builder import build_query
from models.encoders import ImageEncoder, AudioEncoder, TextEncoder
from reasoning.cognitive_analysis import enhanced_cognitive_processing
from reasoning.gemini_flash_reasoner import get_flash_reasoner
from .memory_agent import LessHearableAgent
from sentence_transformers import SentenceTransformer
import torch
from torchvision import transforms
from PIL import Image
import librosa
import os

# Import OutputLayer
from .output_layer import OutputLayer


class NormalPersonAgent(NeuroAdaptiveAgent):
    def __init__(self, memory_manager, img_enc, aud_enc, txt_enc, device,
                 manager=None, less_hearable_agent=None, api_key: str = None,
                 output_layer=None):
        super().__init__(memory_manager)
        self.agent_type = "normal_person"
        self.img_enc = img_enc
        self.aud_enc = aud_enc
        self.txt_enc = txt_enc
        self.device = device
        self.manager = manager

        # Reasoners
        self.vision_reasoner = get_vision_reasoner()
        self.flash_reasoner = get_flash_reasoner(api_key)
        self.caption_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.summary = None

        # LessHearableAgent
        if less_hearable_agent is not None:
            self.less_hearable_agent = less_hearable_agent
        else:
            less_hearable_memory = {}
            self.less_hearable_agent = LessHearableAgent(
                txt_enc=txt_enc,
                memory_store=less_hearable_memory,
                device=device,
                api_key=api_key
            )

        # -------------------------------
        # Reuse OutputLayer passed from AgentManager
        # -------------------------------
        if output_layer is not None:
            self.output_layer = output_layer
        else:
            self.output_layer = None
            print(f"[Warning] No OutputLayer provided from AgentManager.")



    # -------------------------------
    # Image & Audio Loading Helpers
    # -------------------------------
    def load_image(self, path):
        img_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        return img_tf(Image.open(path).convert("RGB")).unsqueeze(0).to(self.device)

    def load_audio(self, path, sr=16000, n_mels=64, target_frames=128):
        y, sr = librosa.load(path, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        t = mel_db.shape[1]
        if t < target_frames:
            mel_db = np.pad(mel_db, ((0, 0), (0, target_frames - t)), mode="constant")
        elif t > target_frames:
            start = (t - target_frames) // 2
            mel_db = mel_db[:, start:start + target_frames]
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
        return torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)


    # -------------------------------
    # Process modalities and generate reasoning
    # -------------------------------
    def process_modalities(self, image_path=None, audio_path=None, text_input=None, top_n=5) -> Dict[str, Any]:
        # query is now a List[Dict[str, Any]] from the updated build_query
        query = build_query(
            image_path=image_path,
            audio_path=audio_path,
            text_input=text_input,
            img_enc=self.img_enc,
            aud_enc=self.aud_enc,
            txt_enc=self.txt_enc,
            device=self.device
        )

        # retriever.retrieve_topn is now updated to accept the list of queries
        results = self.retriever.retrieve_topn(query, top_n=top_n)
        top_match = results[0] if results else {}

        vision_data = {}
        if image_path:
            try:
                # We assume vision_reasoner can handle a single image path and text input
                vision_data = self.vision_reasoner.analyze_image_comprehension(image_path, text_input)
            except Exception:
                vision_data = {}

        # cog_analysis receives the list of queries and the combined retrieval results
        cog_analysis = enhanced_cognitive_processing(query, results, vision_data)

        flash_reasoning = None
        if self.flash_reasoner:
            try:
                # flash_reasoner receives the list of queries
                flash_reasoning = self.flash_reasoner.generate_advanced_reasoning(query, cog_analysis)
            except Exception as e:
                flash_reasoning = f"[Gemini error: {e}]"

        return {
            "top_match": top_match,
            "all_results": results,
            "cognitive_analysis": cog_analysis,
            "vision_data": vision_data,
            "query": query, # The full list of modality queries
            "flash_reasoning": flash_reasoning
        }
    # ... other methods and class definition below ...


    # -------------------------------
    # Generate human-readable reasoning
    # -------------------------------
    def generate_reasoned_sentence(self, processed_data: Dict[str, Any]) -> str:
        flash_reasoning = processed_data.get("flash_reasoning")
        if flash_reasoning and isinstance(flash_reasoning, str):
            return flash_reasoning

        top_match = processed_data.get("top_match", {})
        vision_data = processed_data.get("vision_data", {})
        caption = top_match.get("caption", "")

        objects = self._extract_objects_from_caption(caption)
        manipulation = self._analyze_manipulation_from_text(caption)

        if vision_data:
            objects.extend([obj['class'] for obj in vision_data.get('objects_detected', [])])
            manipulation['held_objects'].extend(
                vision_data.get('manipulation_analysis', {}).get('held_objects', [])
            )

        objects = list(set(objects))
        manipulation['held_objects'] = list(set(manipulation['held_objects']))

        if manipulation['held_objects']:
            reasoned_sentence = (
                f"The person is actively manipulating {', '.join(manipulation['held_objects'])} "
                f"while interacting with {', '.join(objects[:3])}. "
            )
        else:
            reasoned_sentence = f"The scene contains {', '.join(objects[:3])}. "

        reasoned_sentence += f"with {top_match.get('sim_value', 0):.0%} visual similarity to previous experiences."
        return reasoned_sentence


    # -------------------------------
    # Full pipeline: process → forward → LessHearableAgent
    # -------------------------------
    def process_and_forward(self, image_path=None, audio_path=None, text_input=None, top_n=5) -> Dict[str, Any]:
        # Step 1: Process all modalities
        processed = self.process_modalities(image_path, audio_path, text_input, top_n)
        caption = self.generate_reasoned_sentence(processed)

        # Step 2: Encode caption
        caption_embedding = self.caption_encoder.encode(caption, convert_to_numpy=True).flatten().tolist()

        # Step 3: Create query_key from image if available
        query_key = None
        if image_path:
            tensor = self.load_image(image_path)
            with torch.no_grad():
                img_embedding = self.img_enc(tensor).cpu().numpy().flatten().tolist()
            query_key = img_embedding
        elif audio_path:
            tensor = self.load_audio(audio_path)
            with torch.no_grad():
                aud_embedding = self.aud_enc(tensor).cpu().numpy().flatten().tolist()
            query_key = aud_embedding
        

        # Step 4: Store in LessHearableAgent
        summary = None
        if self.less_hearable_agent:
            record, summary = self.less_hearable_agent.summarize_and_store(
                query_key=query_key,
                reasoned_sentence=caption,
                value_vec=caption_embedding
            )
            self.summary = summary
            print("summary:", summary)


        # Step 5: Retrieve structured facts/knowledge using OutputLayer
        knowledge_output = None
        if hasattr(self, "output_layer") and self.output_layer is not None:
            query_text = self.summary if self.summary else caption
            knowledge_output = self.output_layer_run(query_text=query_text)

        # Step 6: Return all results
        return {
            "reasoned_sentence": caption,
            "caption_embedding": caption_embedding,
            "query_key": query_key,
            "less_hearable_summary": summary,
            "knowledge_result": knowledge_output
        }


    # -------------------------------
    # Use OutputLayer for structured facts + knowledge
    # -------------------------------
    def output_layer_run(self, query_text: str = None, top_n: int = 3) -> Dict[str, Any]:
        if not self.output_layer:
            return {"error": "OutputLayer not initialized"}

        if query_text is None:
            query_text = self.summary if self.summary else "No query available"

        knowledge_result, pretty_out = self.output_layer.run(query_text, top_n=top_n)
        return {
            "query": query_text,
            "knowledge_result": knowledge_result,
            "pretty_output": pretty_out
        }


    # -------------------------------
    # Helper functions
    # -------------------------------
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
