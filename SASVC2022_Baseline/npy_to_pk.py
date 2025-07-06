import os
import pickle as pk
import numpy as np

def convert_cm_npy_to_pk(npy_dir, output_path):
    cm_emb_dic = {}

    for filename in os.listdir(npy_dir):
        if filename.endswith("_prosody.npy"):
            utt_id = filename.replace("_prosody.npy", "")
            file_path = os.path.join(npy_dir, filename)
            embedding = np.load(file_path)
            cm_emb_dic[utt_id] = embedding


    with open(output_path, "wb") as f:
        pk.dump(cm_emb_dic, f)
    print(f"Saved PK file to: {output_path}")

# Example usage:
set_name = "trn"  # or "dev"
npy_dir = "cm_embedding/train"  # replace with your directory
output_file = f"embeddings/cm_embd_{set_name}.pk"

convert_cm_npy_to_pk(npy_dir, output_file)
