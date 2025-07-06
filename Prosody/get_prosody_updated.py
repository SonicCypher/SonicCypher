import os
import numpy as np
from tqdm import tqdm  # For progress bar
from ref_encoder import reference_encoder  # Import reference encoder function
from hyperparams import Hyperparams as hp

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Path to the saved model checkpoint
CHECKPOINT_PATH = "./logdir/ref_encoder.ckpt-5000"

def process_and_generate_prosody(input_mel_dir, output_prosody_dir, batch_size=16):
    os.makedirs(output_prosody_dir, exist_ok=True)
    print("Entered process_and_generate_prosody")

    processed_files = []
    failed_files = []

    # Reset graph before restoring weights
    tf.reset_default_graph()

    # Define input placeholder (assuming 80 mel bins)
    mel_input = tf.placeholder(dtype=tf.float32, shape=[None, None, 80, 1], name="mel_input")  # Batch size is dynamic

    # Build the model
    prosody_embedding = reference_encoder(mel_input, is_training=False)

    with tf.Session() as sess:
        # Handle missing weights: Initialize from checkpoint while ignoring missing keys
        vars_to_restore = tf.global_variables()
        reader = tf.train.NewCheckpointReader(CHECKPOINT_PATH)
        available_vars = reader.get_variable_to_shape_map()

        assignment_map = {}
        for var in vars_to_restore:
            var_name = var.name.split(':')[0]
            if var_name in available_vars:
                assignment_map[var_name] = var

        tf.train.init_from_checkpoint(CHECKPOINT_PATH, assignment_map)

        # Initialize missing variables
        sess.run(tf.global_variables_initializer())

        print(f"Loading model from {CHECKPOINT_PATH}...")
        print("Model loaded successfully!")

        # Get all mel files
        mel_files = [f for f in os.listdir(input_mel_dir) if f.endswith("_mel.npy")]

        # Process files in batches
        with tqdm(total=len(mel_files), desc="Processing Mel Files", unit="file") as pbar:
            for i in range(0, len(mel_files), batch_size):
                batch_files = mel_files[i:i + batch_size]
                batch_mels = []
                batch_file_names = []

                max_time_dim = 0  # Track the maximum time dimension in the batch

                for file_name in batch_files:
                    try:
                        mel_path = os.path.join(input_mel_dir, file_name)
                        mel = np.load(mel_path)  # Load mel spectrogram

                        # Ensure mel has 80 mel bins
                        if mel.shape[1] < 80:
                            mel = np.pad(mel, ((0, 0), (0, 80 - mel.shape[1])), mode='constant')
                        elif mel.shape[1] > 80:
                            mel = mel[:, :80]

                        # Update max_time_dim
                        max_time_dim = max(max_time_dim, mel.shape[0])

                        # Reshape mel spectrogram to match input shape (N, T, 80, C)
                        mel_4d = mel.reshape(1, mel.shape[0], mel.shape[1], 1)
                        batch_mels.append(mel_4d)
                        batch_file_names.append(file_name)
                    except Exception as e:
                        failed_files.append(file_name)
                        print(f"Error processing {file_name}: {e}")

                if batch_mels:
                    # Pad all mel spectrograms in the batch to the same time dimension
                    padded_batch_mels = []
                    for mel in batch_mels:
                        time_dim = mel.shape[1]
                        if time_dim < max_time_dim:
                            padding = ((0, 0), (0, max_time_dim - time_dim), (0, 0), (0, 0))
                            mel = np.pad(mel, padding, mode='constant')
                        padded_batch_mels.append(mel)

                    # Stack batch mels into a single numpy array
                    batch_mels = np.vstack(padded_batch_mels)

                    # Run session to get prosody embeddings
                    try:
                        prosody_embeddings_np = sess.run(prosody_embedding, feed_dict={mel_input: batch_mels})

                        # Save prosody embeddings
                        for j, file_name in enumerate(batch_file_names):
                            prosody_path = os.path.join(output_prosody_dir, file_name.replace("_mel.npy", "_prosody.npy"))
                            np.save(prosody_path, prosody_embeddings_np[j])
                            processed_files.append(file_name)
                    except Exception as e:
                        for file_name in batch_file_names:
                            failed_files.append(file_name)
                        print(f"Error processing batch: {e}")

                # Update progress bar
                pbar.update(len(batch_files))

    # Summary
    print("\n--- Processing Summary ---")
    print(f"Total files processed successfully: {len(processed_files)}")
    print(f"Total files failed: {len(failed_files)}")
    if failed_files:
        print("Failed files:", failed_files)

# Define directories
input_mel_dir = "./mels_dev"
output_prosody_dir = "./prosody_embeddings_dev"

# Run the function
process_and_generate_prosody(input_mel_dir, output_prosody_dir, batch_size=16)