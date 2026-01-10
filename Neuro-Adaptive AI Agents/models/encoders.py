import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from config.settings import DEVICE

class ImageEncoder(nn.Module):
    def __init__(self, out_dim=15):
        super().__init__()
        base = models.resnet18(weights=None)  # We'll load pretrained weights from your file
        feat_dim = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        h = self.backbone(x)
        z = F.normalize(self.proj(h), dim=-1)
        return z

class AudioEncoder(nn.Module):
    def __init__(self, out_dim=15):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.proj = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        h = self.conv(x).flatten(1)
        z = F.normalize(self.proj(h), dim=-1)
        return z

class TextEncoder(nn.Module):
    def __init__(self, out_dim=15):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        self.sbert = SentenceTransformer("all-MiniLM-L6-v2")
        in_dim = 384
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_dim)
        )
        
        # Freeze the SBERT model
        for param in self.sbert.parameters():
            param.requires_grad = False

    def forward(self, texts):
        # Handle both single string and list of strings
        if isinstance(texts, str):
            texts = [texts]
        
        with torch.no_grad():
            emb = self.sbert.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
        
        emb = torch.tensor(emb, dtype=torch.float32, device=next(self.proj.parameters()).device)
        z = F.normalize(self.proj(emb), dim=-1)
        
        # If input was single string, return single embedding
        if z.shape[0] == 1:
            return z.squeeze(0)
        return z