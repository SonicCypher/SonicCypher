import os
import numpy as np

def load_all_utterance_ids(txt_files):
    utt_ids = set()
    for file_path in txt_files:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    _, utts_str = line.strip().split()
                    utts = utts_str.split(',')
                    utt_ids.update(utts)
                except ValueError:
                    print(f"[Warning] Skipping malformed line: {line.strip()}")
    return utt_ids

def compute_global_mean_embedding(utt_ids, embedding_dir):
    embeddings = []
    found = 0
    missing = 0
    for utt_id in utt_ids:
        embedding_path = os.path.join(embedding_dir, f"{utt_id}_prosody.npy")
        if os.path.exists(embedding_path):
            emb = np.load(embedding_path)
            embeddings.append(emb)
            found += 1
        else:
            print(f"[Warning] Missing: {embedding_path}")
            missing += 1

    print(f"[Info] Found embeddings: {found}")
    print(f"[Info] Missing embeddings: {missing}")

    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        raise ValueError("No valid embeddings found!")

def save_embedding(embedding, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embedding)
    print(f"[Saved] Mean embedding saved to: {output_path}")

def main():
    # Development set
    embedding_dir_dev = '/home/cse/SonicCypher/Wathmi/SonicCypher/Prosody/prosody_embeddings_dev'
    enrol_dev_female_file = '/home/cse/SonicCypher/Wathmi/SonicCypher/SASVC2022_Baseline/LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.dev.female.trn.txt'
    enrol_dev_male_file = '/home/cse/SonicCypher/Wathmi/SonicCypher/SASVC2022_Baseline/LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.dev.male.trn.txt'
    output_file_dev = 'mean_embedding/mean_prosody_embedding_dev.npy'

    print("\n=== Processing Development Set ===")
    dev_utterance_ids = load_all_utterance_ids([enrol_dev_female_file, enrol_dev_male_file])
    dev_mean_embedding = compute_global_mean_embedding(dev_utterance_ids, embedding_dir_dev)
    save_embedding(dev_mean_embedding, output_file_dev)

    # Evaluation set
    embedding_dir_eval = '/home/cse/SonicCypher/Wathmi/SonicCypher/Prosody/prosody_embeddings_eval'
    enrol_eval_female_file = '/home/cse/SonicCypher/Wathmi/SonicCypher/SASVC2022_Baseline/LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.female.trn.txt'
    enrol_eval_male_file = '/home/cse/SonicCypher/Wathmi/SonicCypher/SASVC2022_Baseline/LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.male.trn.txt'
    output_file_eval = 'mean_embedding/mean_prosody_embedding_eval.npy'

    print("\n=== Processing Evaluation Set ===")
    eval_utterance_ids = load_all_utterance_ids([enrol_eval_female_file, enrol_eval_male_file])
    eval_mean_embedding = compute_global_mean_embedding(eval_utterance_ids, embedding_dir_eval)
    save_embedding(eval_mean_embedding, output_file_eval)

if __name__ == '__main__':
    main()
