import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# Choose device: GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def compute_cosine_similarity(emb1, emb2, normalize=False):
    emb1 = torch.tensor(emb1, dtype=torch.float32, device=device)
    emb2 = torch.tensor(emb2, dtype=torch.float32, device=device)
    if normalize:
        emb1 = F.normalize(emb1, dim=0)
        emb2 = F.normalize(emb2, dim=0)
    return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer * 100

def process_file(file_path, embedding_dir, mean_embedding_np):
    scores_norm = []
    scores_raw = []
    labels = []

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            utt_id = parts[1]
            label = parts[-1]

            emb_file = f"{utt_id}_prosody.npy"
            emb_path = os.path.join(embedding_dir, emb_file)

            if not os.path.exists(emb_path):
                print(f"[Warning] Missing embedding: {emb_path}")
                continue

            emb = np.load(emb_path)

            # Cosine similarity with and without normalization (on GPU)
            score_raw = compute_cosine_similarity(emb, mean_embedding_np, normalize=False)
            score_norm = compute_cosine_similarity(emb, mean_embedding_np, normalize=True)

            scores_raw.append(score_raw)
            scores_norm.append(score_norm)
            labels.append(1 if label == 'target' else 0)

    eer_raw = compute_eer(labels, scores_raw)
    eer_norm = compute_eer(labels, scores_norm)

    return eer_raw, eer_norm

# --- File paths ---
files_to_process = {
    'TTS': '/home/cse/SonicCypher/Wathmi/SonicCypher/Prosody/score_fusion/sasv/tts.txt',
    'Bonafide': '/home/cse/SonicCypher/Wathmi/SonicCypher/Prosody/score_fusion/sasv/bonafide.txt',
    'VC': '/home/cse/SonicCypher/Wathmi/SonicCypher/Prosody/score_fusion/sasv/vc.txt',
    'SASV': '/home/cse/SonicCypher/Wathmi/SonicCypher/SASVC2022_Baseline/LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.gi.trl.txt'
}

embedding_dir = '/home/cse/SonicCypher/Wathmi/SonicCypher/Prosody/prosody_embeddings_eval'
mean_embedding_path = '/home/cse/SonicCypher/Wathmi/SonicCypher/Prosody/score_fusion/mean_embedding/mean_prosody_embedding_eval.npy'

# --- Load mean embedding and move to GPU ---
mean_embedding = np.load(mean_embedding_path)

# --- Compare EERs ---
for name, path in files_to_process.items():
    eer_raw, eer_norm = process_file(path, embedding_dir, mean_embedding)
    print(f"{name} EER (no normalization): {eer_raw:.2f}%")
    print(f"{name} EER (with normalization): {eer_norm:.2f}%\n")
