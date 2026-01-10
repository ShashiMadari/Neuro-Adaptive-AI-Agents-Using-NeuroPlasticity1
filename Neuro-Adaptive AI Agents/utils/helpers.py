import torch
from torchvision import transforms
from PIL import Image
import librosa
import numpy as np

def load_image(path):
    img_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    ])
    return img_tf(Image.open(path).convert("RGB")).unsqueeze(0)

def load_audio(path, sr=16000, n_mels=64, target_frames=128):
    y, sr = librosa.load(path, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    t = mel_db.shape[1]
    if t < target_frames:
        mel_db = np.pad(mel_db, ((0,0),(0,target_frames-t)), mode="constant")
    elif t > target_frames:
        start = (t - target_frames)//2
        mel_db = mel_db[:, start:start+target_frames]
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std()+1e-6)
    return torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)