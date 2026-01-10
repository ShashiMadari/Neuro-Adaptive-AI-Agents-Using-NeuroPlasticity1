# ----------------------------------
# blind_person_agent.py
# ----------------------------------
import numpy as np
from typing import Any, Dict
from .base_agent import NeuroAdaptiveAgent
from core.query_builder import build_query
from reasoning.cognitive_analysis import enhanced_cognitive_processing
from reasoning.gemini_flash_reasoner import get_flash_reasoner
from .memory_agent import LessHearableAgent
from sentence_transformers import SentenceTransformer  # type: ignore
import torch
import librosa  # type: ignore


class BlindPersonAgent(NeuroAdaptiveAgent):
    def __init__(self, memory_manager, aud_enc, txt_enc, device,manager=None,
                 less_hearable_agent=None, api_key: str = None, output_layer=None):
        super().__init__(memory_manager)
        self.agent_type = "blind_person"
        self.aud_enc = aud_enc
        self.txt_enc = txt_enc
        self.device = device
        self.flash_reasoner = get_flash_reasoner(api_key)
        self.caption_encoder = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim
        self.summary = None
        self.manager = manager            # 👈 added
        self.output_layer = output_layer
        # ✅ LessHearableAgent setup
        if less_hearable_agent is not None:
            self.less_hearable_agent = less_hearable_agent
        else:
            self.less_hearable_agent = LessHearableAgent(
                txt_enc=txt_enc,
                memory_store=[],
                device=device,
                api_key=api_key
            )

        # ✅ OutputLayer (facts + knowledge retrieval)
        if output_layer is not None:
            self.output_layer = output_layer
        else:
            self.output_layer = None
            print(f"[Warning] No OutputLayer provided to BlindPersonAgent.")

    # -------------------------------
    # Audio helper
    # -------------------------------
    def load_audio(self, path, sr=16000, n_mels=64, target_frames=128):
        """Convert audio into normalized mel-spectrogram tensor"""
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
    # Modalities processing
    # -------------------------------


    def process_modalities(self, audio_path=None, text_input=None, top_n=5) -> Dict[str, Any]:
        """Blind agent: ignores images, focuses on audio + text"""
        
        # query is a List[Dict[str, Any]] containing only audio and/or text entries
        query = build_query(
            image_path=None,  # ignore images
            audio_path=audio_path,
            text_input=text_input,
            img_enc=None,
            aud_enc=self.aud_enc,
            txt_enc=self.txt_enc,
            device=self.device
        )

        # retrieve_topn handles the list of queries and returns combined results
        results = self.retriever.retrieve_topn(query, top_n=top_n)
        
        # enhanced_cognitive_processing is expected to handle query as a list
        cog_analysis = enhanced_cognitive_processing(query, results, vision_data={})

        flash_reasoning = None
        if self.flash_reasoner:
            try:
                # generate_advanced_reasoning is expected to handle query as a list
                flash_reasoning = self.flash_reasoner.generate_advanced_reasoning(query, cog_analysis)
            except Exception as e:
                flash_reasoning = f"[Gemini error: {e}]"

        top_match = results[0] if results else {}
        return {
            "top_match": top_match,
            "all_results": results,
            "cognitive_analysis": cog_analysis,
            "query": query, # Returns the list of query entries
            "flash_reasoning": flash_reasoning
        }


    # -------------------------------
    # Reason sentence
    # -------------------------------
    def generate_reasoned_sentence(self, processed_data: Dict[str, Any]) -> str:
        """Prefer Gemini output, fallback to audio reasoning"""
        flash_reasoning = processed_data.get("flash_reasoning")
        if flash_reasoning and isinstance(flash_reasoning, str):
            return flash_reasoning

        caption = processed_data.get("top_match", {}).get("caption", "")
        return f"Audio-centric reasoning: {caption}" if caption else "No clear audio reasoning available."

    # -------------------------------
    # Full pipeline: process → forward → LessHearableAgent → OutputLayer
    # -------------------------------
    def process_and_forward(self, image_path=None,audio_path=None, text_input=None, top_n=5) -> Dict[str, Any]:
        # Step 1: Process audio/text
        processed = self.process_modalities(audio_path=audio_path, text_input=text_input, top_n=top_n)
        caption = self.generate_reasoned_sentence(processed)

        # Step 2: Caption → embedding
        caption_embedding = self.caption_encoder.encode(caption, convert_to_numpy=True).flatten().tolist()

        # Step 3: Use audio embedding as query_key
        if audio_path:
            tensor = self.load_audio(audio_path)
            with torch.no_grad():
                aud_embedding = self.aud_enc(tensor).cpu().numpy().flatten().tolist()
            query_key = aud_embedding
        else:
            query_key = caption_embedding

        # Step 4: Memory update
        summary = None
        if self.less_hearable_agent:
            record, summary = self.less_hearable_agent.summarize_and_store(
                query_key=query_key,
                reasoned_sentence=caption,
                value_vec=caption_embedding
            )
            self.summary = summary
            print("summary:", summary)

        # Step 5: Knowledge retrieval via OutputLayer
        knowledge_output = None
        if hasattr(self, "output_layer") and self.output_layer is not None:
            query_text = self.summary if self.summary else caption
            knowledge_output = self.output_layer_run(query_text=query_text)

        # Step 6: Return all
        return {
            "reasoned_sentence": caption,
            "caption_embedding": caption_embedding,
            "query_key": query_key,
            "less_hearable_summary": summary,
            "knowledge_result": knowledge_output
        }

    # -------------------------------
    # OutputLayer usage
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
