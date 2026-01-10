from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config.settings import DEVICE, NLI_MODEL
import torch
class NLIModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(DEVICE)
        self.model.eval()
    
    def entailment_score(self, premise: str, hypothesis: str) -> float:
        inputs = self.tokenizer(
            premise, hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(DEVICE)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()
        return float(probs[2])  # entailment probability