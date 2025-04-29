import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Paths to directories
flac_dir = r"E:\Semester 8\FYP\SonicCypher\ASVspoof 2019\ASVspoof2019_LA_train\flac"
embedding_dir = r"E:\Semester 8\FYP\SonicCypher\prosody_embeddings"

# Get a list of all .flac files in the directory
flac_files = [f for f in os.listdir(flac_dir) if f.endswith(".flac")]

# Process only the first 20 .flac files
num_files_to_process = min(20, len(flac_files))
flac_files = flac_files[:num_files_to_process]

for flac_file in flac_files:
    flac_path = os.path.join(flac_dir, flac_file)
    print(f"\nProcessing file: {flac_file}")

    # Load the audio file
    y, sr = librosa.load(flac_path, sr=None)

    # Extract pitch (F0) and energy
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=500)
    energy = librosa.feature.rms(y=y)[0]

    # Get corresponding embedding file name
    embedding_filename = flac_file.replace(".flac", "_prosody.npy")
    embedding_path = os.path.join(embedding_dir, embedding_filename)

    # Load prosody embedding if available
    if os.path.exists(embedding_path):
        embedding = np.load(embedding_path)
        print("Prosody Embedding Shape:", embedding.shape)
    else:
        print(f"Prosody embedding not found: {embedding_path}")
        embedding = None

    # Visualize Pitch and Energy (F0, Energy) from Original Audio
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(f0, label="Original Pitch (F0)", color='b')
    plt.title(f"Original Prosody Features for {flac_file}")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(energy, label="Original Energy", color='r')
    plt.title("Original Energy Feature")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # If embedding exists, compare the prosody embedding
    if embedding is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(embedding[0], label="Generated Prosody Embedding", color='g')
        plt.title(f"Generated Prosody Embedding for {flac_file}")
        plt.legend()
        plt.show()

print("\nProcessing Complete!")
