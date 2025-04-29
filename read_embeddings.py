import os
import numpy as np

# Directory containing embeddings
embedding_dir = r"E:\Semester 8\FYP\SonicCypher\prosody_embeddings"

# List all .npy files
for file_name in os.listdir(embedding_dir):
    if file_name.endswith(".npy"):
        file_path = os.path.join(embedding_dir, file_name)
        
        # Load embedding
        embedding = np.load(file_path)
        
        # Print details
        print(f"File: {file_name}")
        print(f"Shape: {embedding.shape}")
        print(f"Contents:\n{embedding}\n")
