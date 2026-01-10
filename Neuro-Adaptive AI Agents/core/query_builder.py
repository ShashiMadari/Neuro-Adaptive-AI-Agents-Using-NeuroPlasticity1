import torch
import numpy as np
from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer

# Import processors (assuming these exist in your project structure)
from processors.image_processor import process_image
from processors.audio_processor import process_audio
from processors.text_processor import process_text

# Import models (assuming these exist in your project structure)
from models.encoders import ImageEncoder, AudioEncoder, TextEncoder

# Initialize sentence transformer
# NOTE: This should ideally be loaded once at system startup to prevent repeated I/O.
sem_model = SentenceTransformer("all-MiniLM-L6-v2")

def build_query(image_path=None, audio_path=None, text_input=None, 
                img_enc=None, aud_enc=None, txt_enc=None, device=None) -> List[Dict[str, Any]]:
    """
    Builds a list of query entries, processing and encoding all available modalities 
    (image, audio, text) into separate, structured dictionaries.

    Args:
        image_path (str, optional): Path to the image file.
        audio_path (str, optional): Path to the audio file.
        text_input (str, optional): Raw text input.
        img_enc (ImageEncoder, optional): Image feature encoder.
        aud_enc (AudioEncoder, optional): Audio feature encoder.
        txt_enc (TextEncoder, optional): Text feature encoder.
        device (torch.device, optional): The device to run computations on.

    Returns:
        List[Dict[str, Any]]: A list containing a dictionary for each successfully 
                              processed modality.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize a list to hold all query dictionaries
    queries: List[Dict[str, Any]] = []

    # --- 1. Image Modality ---
    if image_path and img_enc:
        try:
            # Placeholder for the entry dictionary
            entry: Dict[str, Any] = {}
            
            proc = process_image(image_path)
            from utils.helpers import load_image
            tensor = load_image(image_path).to(device)
            
            with torch.no_grad():
                key = img_enc(tensor).cpu().numpy().flatten().tolist()
            
            # Use the image caption for the semantic value encoding
            value = sem_model.encode(proc["caption"]).tolist()
            
            entry.update({
                "type": "image",
                "query_text": proc["caption"],
                "key": key, # Feature vector from ImageEncoder
                "value": value, # Semantic vector from SentenceTransformer
                "meta": {
                    "emotion": proc.get("emotion", ""),
                    "importance": proc.get("importance", 0.0),
                    "caption": proc.get("caption", "")
                }
            })
            queries.append(entry)
        except Exception as e:
            print(f"Error building image query: {e}")
            # Continue to the next modality even if this one fails

    # --- 2. Audio Modality ---
    if audio_path and aud_enc:
        try:
            # Placeholder for the entry dictionary
            entry: Dict[str, Any] = {}
            
            proc = process_audio(audio_path)
            from utils.helpers import load_audio
            tensor = load_audio(audio_path).to(device)
            
            with torch.no_grad():
                key = aud_enc(tensor).cpu().numpy().flatten().tolist()
            
            # Use the transcribed text for the semantic value encoding
            transcribed_text = proc.get("transcribed", "") or "non-speech"
            value = sem_model.encode(transcribed_text).tolist()
            
            entry.update({
                "type": "audio",
                "query_text": proc.get("audio_text", transcribed_text),
                "key": key, # Feature vector from AudioEncoder
                "value": value, # Semantic vector from SentenceTransformer
                "meta": {
                    "emotion": proc.get("emotion", ""),
                    "importance": proc.get("importance", 0.0),
                    "transcribed": transcribed_text
                }
            })
            queries.append(entry)
        except Exception as e:
            print(f"Error building audio query: {e}")
            # Continue to the next modality even if this one fails

    # --- 3. Text Modality ---
    if text_input and txt_enc:
        try:
            # Placeholder for the entry dictionary
            entry: Dict[str, Any] = {}
            
            proc = process_text(text_input)
            
            with torch.no_grad():
                # Text encoder now handles single strings properly
                key_tensor = txt_enc(text_input)
                key = key_tensor.cpu().numpy().flatten().tolist()
                
            value = sem_model.encode(text_input).tolist()
            
            entry.update({
                "type": "text",
                "query_text": proc.get("text_summary", text_input),
                "key": key, # Feature vector from TextEncoder
                "value": value, # Semantic vector from SentenceTransformer
                "meta": {
                    "emotion": proc.get("emotion", ""),
                    "importance": proc.get("importance", 0.0),
                    "raw_text": text_input
                }
            })
            queries.append(entry)
        except Exception as e:
            print(f"Error building text query: {e}")
            # Continue to the next modality even if this one fails

    # Return the list of all successful queries
    return queries
