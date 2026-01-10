import numpy as np
from typing import List, Dict, Any

def fuse_embeddings(keys: List[np.ndarray], values: List[np.ndarray]) -> Dict[str, Any]:
    """
    Fuse multiple modality embeddings into a single query representation.
    """
    fused_key = np.mean(keys, axis=0).tolist()
    fused_value = np.mean(values, axis=0).tolist()
    return {"key": fused_key, "value": fused_value}

def build_multimodal_query(image_path=None, audio_path=None, text_input=None, 
                          img_enc=None, aud_enc=None, txt_enc=None, device=None) -> Dict[str, Any]:
    """
    Build a multimodal query entry without circular imports
    """
    from core.query_builder import build_query
    
    subqueries = []
    
    print(f"Processing modalities - Image: {image_path}, Audio: {audio_path}, Text: {text_input}")
    
    if image_path and img_enc:
        try:
            img_query = build_query(image_path=image_path, img_enc=img_enc, device=device)
            print(f"Image query keys: {list(img_query.keys()) if img_query else 'None'}")
            if img_query and 'key' in img_query:
                subqueries.append(img_query)
            else:
                print("Warning: Image query missing 'key' field")
        except Exception as e:
            print(f"Error processing image: {e}")

    if audio_path and aud_enc:  # Added aud_enc check
        try:
            aud_query = build_query(audio_path=audio_path, aud_enc=aud_enc, device=device)
            print(f"Audio query keys: {list(aud_query.keys()) if aud_query else 'None'}")
            if aud_query and 'key' in aud_query:
                subqueries.append(aud_query)
            else:
                print("Warning: Audio query missing 'key' field")
        except Exception as e:
            print(f"Error processing audio: {e}")

    if text_input and txt_enc:  # Added txt_enc check
        try:
            txt_query = build_query(text_input=text_input, txt_enc=txt_enc, device=device)
            print(f"Text query keys: {list(txt_query.keys()) if txt_query else 'None'}")
            if txt_query and 'key' in txt_query:
                subqueries.append(txt_query)
            else:
                print("Warning: Text query missing 'key' field")
        except Exception as e:
            print(f"Error processing text: {e}")

    print(f"Final subqueries count: {len(subqueries)}")
    
    if not subqueries:
        raise ValueError("No input modalities provided.")

    if len(subqueries) == 1:
        return subqueries[0]

    # Debug: check each subquery has required fields
    for i, sq in enumerate(subqueries):
        print(f"Subquery {i}: {list(sq.keys())}")
        if 'key' not in sq:
            print(f"ERROR: Subquery {i} missing 'key': {sq}")
        if 'value' not in sq:
            print(f"ERROR: Subquery {i} missing 'value': {sq}")

    fused = fuse_embeddings(
        keys=[sq["key"] for sq in subqueries],
        values=[sq["value"] for sq in subqueries]
    )

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