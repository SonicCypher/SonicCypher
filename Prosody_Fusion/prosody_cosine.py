import os
import torch
import torch.nn.functional as F
import numpy as np

def add_cosine_scores(txt_path, embedding_dir, mean_embedding_path, output_path):
    # Load mean embedding
    mean_embedding_np = np.load(mean_embedding_path)
    mean_embedding = torch.from_numpy(mean_embedding_np).float().unsqueeze(0)

    # Process each line and compute cosine similarity
    with open(txt_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            parts = line.strip().split()
            if len(parts) < 4:
                f_out.write(line)
                continue
            utt_id = parts[1]
            emb_file = os.path.join(embedding_dir, f'{utt_id}_prosody.npy')
            if not os.path.exists(emb_file):
                print(f'Warning: {utt_id} embedding not found. Writing 0.0 similarity.')
                cosine_sim = 0.0
            else:
                utt_embedding_np = np.load(emb_file)
                utt_embedding = torch.from_numpy(utt_embedding_np).float().unsqueeze(0)
                cosine_sim = F.cosine_similarity(mean_embedding, utt_embedding, dim=1).item()

            # Append similarity to line
            f_out.write(line.strip() + f' {cosine_sim:.6f}\n')

    print(f'Done! Saved: {output_path}')

# === DEV SET ===
dev_txt = '/home/cse/SonicCypher/Wathmi/SonicCypher/SASVC2022_Baseline/LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.dev.gi.trl.txt'
dev_embedding_dir = '/home/cse/SonicCypher/Wathmi/SonicCypher/Prosody/prosody_embeddings_dev'
dev_mean = '/home/cse/SonicCypher/Wathmi/SonicCypher/Prosody/score_fusion/mean_embedding/mean_prosody_embedding_dev.npy'
dev_output = os.path.join(
    '/home/cse/SonicCypher/Wathmi/SonicCypher/Prosody/score_fusion',
    'ASVspoof2019.LA.asv.dev.trl_with_cosine.txt'
)

add_cosine_scores(dev_txt, dev_embedding_dir, dev_mean, dev_output)

# === TEST SET ===
test_txt = '/home/cse/SonicCypher/Wathmi/SonicCypher/SASVC2022_Baseline/LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.gi.trl.txt'
test_embedding_dir = '/home/cse/SonicCypher/Wathmi/SonicCypher/Prosody/prosody_embeddings_eval'
test_mean = '/home/cse/SonicCypher/Wathmi/SonicCypher/Prosody/score_fusion/mean_embedding/mean_prosody_embedding_eval.npy'
test_output = os.path.join(
    '/home/cse/SonicCypher/Wathmi/SonicCypher/Prosody/score_fusion',
    'ASVspoof2019.LA.asv.eval.trl_with_cosine.txt'
)

add_cosine_scores(test_txt, test_embedding_dir, test_mean, test_output)