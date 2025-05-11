import os
import pickle as pk
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar

def convert_cm_npy_to_pk(npy_dir, output_path):
    cm_emb_dic = {}

    # Get the list of all files in the directory
    file_list = [f for f in os.listdir(npy_dir) if f.endswith("_prosody.npy")]

    # Use tqdm to show progress
    with tqdm(total=len(file_list), desc="Processing Files", unit="file") as pbar:
        for filename in file_list:
            utt_id = filename.replace("_prosody.npy", "")
            file_path = os.path.join(npy_dir, filename)
            embedding = np.load(file_path)
            cm_emb_dic[utt_id] = embedding

            # Update the progress bar
            pbar.update(1)

    # Save the dictionary as a pickle file
    with open(output_path, "wb") as f:
        pk.dump(cm_emb_dic, f)
    print(f"Saved PK file to: {output_path}")

# Example usage:
set_name = "trn"  # or "dev"
npy_dir = "cm_embedding/train"  # replace with your directory
output_file = f"embeddings/cm_embd_{set_name}.pk"

convert_cm_npy_to_pk(npy_dir, output_file)