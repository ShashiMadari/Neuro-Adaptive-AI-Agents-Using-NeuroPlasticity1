# models/reasoner.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # Changed back to Seq2SeqLM
from config.settings import REASONER_MODEL, DEVICE
import torch

class Reasoner:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(REASONER_MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(REASONER_MODEL).to(DEVICE)
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self, prompt, max_new_tokens=256):
        # For T5 models, use simple prompt formatting
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(DEVICE)
        
        # Generate response
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response