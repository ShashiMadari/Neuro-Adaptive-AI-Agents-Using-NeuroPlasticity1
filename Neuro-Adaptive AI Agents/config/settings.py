import torch

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model paths
REASONER_MODEL = "google/flan-t5-large"
NLI_MODEL = "facebook/bart-large-mnli"
SENTENCE_TRANSFORMER = "all-MiniLM-L6-v2"

# Memory settings
MEMORY_FILE = "memory_store/working_memory.json"
TOP_N = 5

# Retrieval parameters
EMBED_DIM = 15
ALPHA = 0.20
MODALITY_BOOST = {
    "image": 0.03,
    "text": 0.03,
    "audio": 0.02
}
KEYWORD_BOOST = 0.12

# Processing parameters
AUDIO_SR = 16000
AUDIO_N_MELS = 64
AUDIO_TARGET_FRAMES = 128