# multimodal_train.py
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import librosa
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
EMBED_DIM = 15
BATCH_SIZE = 64
LR = 3e-4
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEMPERATURE = 0.07
TARGET_FRAMES = 128  # fixed time frames for log-mel
N_MELS = 64
SR = 16000
DATASET_DIR = r"C:\Users\shash\OneDrive\Desktop\Main-project\dataset_triplets"
NUM_WORKERS = 0 if os.name == "nt" else 4

# -----------------------------
# Triplet loader
# -----------------------------
def load_triplets_from_dir(base_dir, max_items=None):
    triplets = []
    for folder in sorted(os.listdir(base_dir)):
        fpath = os.path.join(base_dir, folder)
        if not os.path.isdir(fpath):
            continue
        img = os.path.join(fpath, "image.jpg")
        aud = os.path.join(fpath, "audio.wav")
        cap = os.path.join(fpath, "caption.txt")

        if os.path.exists(img) and os.path.exists(aud) and os.path.exists(cap):
            try:
                with open(cap, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                if len(text) == 0:
                    print(f"⚠️ Empty caption in {folder}, skipping")
                    continue
                triplets.append({"image": img, "text": text, "audio": aud})
            except Exception as e:
                print(f"⚠️ Error reading caption in {folder}: {e}")
        else:
            print(f"⚠️ Missing files in {folder}, skipping...")

        if max_items and len(triplets) >= max_items:
            break

    print(f"✅ Collected {len(triplets)} triplets from {base_dir}")
    return triplets

# -----------------------------
# Dataset
# -----------------------------
class TripletDataset(Dataset):
    def __init__(self, triplets):
        self.items = triplets
        self.img_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
        ])

    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        return self.img_tf(img)

    def _load_audio_logmel(self, path):
        y, sr = librosa.load(path, sr=SR)
        if y.shape[0] == 0:
            y = np.zeros(1, dtype=np.float32)
        mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=1024, hop_length=512, n_mels=N_MELS)
        # ✅ FIX: correct ref
        mel_db = librosa.power_to_db(mel, ref=np.max)
        t = mel_db.shape[1]
        if t < TARGET_FRAMES:
            pad = TARGET_FRAMES - t
            mel_db = np.pad(mel_db, ((0,0),(0,pad)), mode="constant")
        elif t > TARGET_FRAMES:
            start = (t - TARGET_FRAMES) // 2
            mel_db = mel_db[:, start:start+TARGET_FRAMES]
        m = mel_db.mean()
        s = mel_db.std() + 1e-6
        mel_db = (mel_db - m) / s
        return torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        item = self.items[i]
        img = self._load_image(item["image"])
        txt = item["text"]
        aud = self._load_audio_logmel(item["audio"])
        return img, txt, aud

# -----------------------------
# Encoders
# -----------------------------
class ImageEncoder(nn.Module):
    def __init__(self, out_dim=EMBED_DIM):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
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

class TextEncoder(nn.Module):
    def __init__(self, out_dim=EMBED_DIM, model_name="all-MiniLM-L6-v2"):
        super().__init__()
        self.sbert = SentenceTransformer(model_name)
        in_dim = 384
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_dim)
        )

    def forward(self, texts):
        with torch.no_grad():
            emb = self.sbert.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
        emb = torch.tensor(emb, dtype=torch.float32, device=DEVICE)
        z = F.normalize(self.proj(emb), dim=-1)
        return z

class AudioEncoder(nn.Module):
    def __init__(self, out_dim=EMBED_DIM):
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

# -----------------------------
# Losses
# -----------------------------
def pairwise_contrastive_loss(z_a, z_b, temperature=TEMPERATURE):
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    logits = (z_a @ z_b.t()) / temperature
    labels = torch.arange(z_a.size(0), device=z_a.device)
    loss_ab = F.cross_entropy(logits, labels)
    loss_ba = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_ab + loss_ba)

def multimodal_loss(z_img, z_txt, z_aud):
    loss_it = pairwise_contrastive_loss(z_img, z_txt)
    loss_ia = pairwise_contrastive_loss(z_img, z_aud)
    loss_ta = pairwise_contrastive_loss(z_txt, z_aud)
    total = loss_it + loss_ia + loss_ta
    return total, {"it": loss_it.item(), "ia": loss_ia.item(), "ta": loss_ta.item()}

@torch.no_grad()
def alignment_top1(z_a, z_b):
    sims = z_a @ z_b.t()
    preds = sims.argmax(dim=1)
    labels = torch.arange(z_a.size(0), device=z_a.device)
    return (preds == labels).float().mean().item()

# -----------------------------
# Training loop
# -----------------------------
def train(triplets, epochs=EPOCHS):
    ds = TripletDataset(triplets)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=(DEVICE!="cpu"))

    img_enc = ImageEncoder().to(DEVICE)
    txt_enc = TextEncoder().to(DEVICE)
    aud_enc = AudioEncoder().to(DEVICE)

    params = list(img_enc.parameters()) + list(txt_enc.proj.parameters()) + list(aud_enc.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=="cuda"))

    for epoch in range(1, epochs+1):
        img_enc.train(); txt_enc.train(); aud_enc.train()
        running = 0.0
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        for imgs, texts, auds in pbar:
            imgs = imgs.to(DEVICE, non_blocking=True)
            auds = auds.to(DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
                z_img = img_enc(imgs)
                z_txt = txt_enc(texts)
                z_aud = aud_enc(auds)
                loss, parts = multimodal_loss(z_img, z_txt, z_aud)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()

            running += loss.item()
            avg_loss = running / (pbar.n + 1)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", **{k: f"{v:.3f}" for k,v in parts.items()}})

        # Eval
        img_enc.eval(); txt_enc.eval(); aud_enc.eval()
        with torch.no_grad():
            try:
                imgs, texts, auds = next(iter(dl))
                imgs = imgs.to(DEVICE); auds = auds.to(DEVICE)
                zi = img_enc(imgs); zt = txt_enc(texts); za = aud_enc(auds)
                acc_it = alignment_top1(zi, zt)
                acc_ia = alignment_top1(zi, za)
                acc_ta = alignment_top1(zt, za)
            except StopIteration:
                acc_it = acc_ia = acc_ta = 0.0

        print(f"Epoch {epoch:02d} | avg_loss {running/len(dl):.4f} | "
              f"it {parts['it']:.4f} ia {parts['ia']:.4f} ta {parts['ta']:.4f} | "
              f"top1 it {acc_it:.3f} ia {acc_ia:.3f} ta {acc_ta:.3f}")

    # Save encoders
    os.makedirs("models", exist_ok=True)
    torch.save(img_enc.state_dict(), "models/img_encoder_15d.pt")
    torch.save(txt_enc.state_dict(), "models/txt_encoder_15d.pt")
    torch.save(aud_enc.state_dict(), "models/aud_encoder_15d.pt")
    print("✅ Saved encoders to models/*.pt")

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    triplets = load_triplets_from_dir(DATASET_DIR, max_items=1000)
    if len(triplets) < 2:
        raise SystemExit("Need at least 2 triplets to train.")
    random.shuffle(triplets)
    if len(triplets) > 1000:
        triplets = triplets[:1000]
    print(f"Starting training on {len(triplets)} triplets, device={DEVICE}")
    train(triplets, epochs=EPOCHS)
