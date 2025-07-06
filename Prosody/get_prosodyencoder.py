import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
# import tensorflow as tf  # Pure TensorFlow 1.x

from ref_encoder import reference_encoder  # Import reference encoder function
from hyperparams import Hyperparams as hp

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Path to the saved model checkpoint
CHECKPOINT_PATH = "./logdir/ref_encoder.ckpt-5000"
# "ref_encoder_updated.ckpt"

def process_and_generate_prosody(input_mel_dir, output_prosody_dir):
    os.makedirs(output_prosody_dir, exist_ok=True)
    print("entered process_and_generate_prosody")

    processed_files = []
    failed_files = []
    
    # Reset graph before restoring weights
    tf.reset_default_graph()

    # Define input placeholder (assuming 80 mel bins)
    mel_input = tf.placeholder(dtype=tf.float32, shape=[1, None, 80, 1], name="mel_input")

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

        for file_name in os.listdir(input_mel_dir):
            print(f"file_name: {file_name}")
            if file_name.endswith("_mel.npy"):
                print(f"Processing {file_name}...")

                try:
                    mel_path = os.path.join(input_mel_dir, file_name)
                    mel = np.load(mel_path)  # Load mel spectrogram

                    # Ensure mel has 80 mel bins
                    if mel.shape[1] < 80:
                        mel = np.pad(mel, ((0, 0), (0, 80 - mel.shape[1])), mode='constant')
                    elif mel.shape[1] > 80:
                        mel = mel[:, :80]

                    # Reshape mel spectrogram to match input shape (N, T, 80, C)
                    mel_4d = mel.reshape(1, mel.shape[0], mel.shape[1], 1)
                    print(f"mel_4d shape: {mel_4d.shape}")

                    audio_name = file_name.replace("_mel.npy", "")
                    print(f"Processing {audio_name}...")

                    # Run session to get prosody embedding
                    prosody_embedding_np = sess.run(prosody_embedding, feed_dict={mel_input: mel_4d})

                    # Save prosody embedding
                    prosody_path = os.path.join(output_prosody_dir, file_name.replace("_mel.npy", "_prosody.npy"))
                    np.save(prosody_path, prosody_embedding_np)

                    processed_files.append(file_name)
                    print(f"Finished processing {file_name}, prosody saved at {prosody_path}.")
                except Exception as e:
                    failed_files.append(file_name)
                    print(f"Error processing {file_name}: {e}")


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
process_and_generate_prosody(input_mel_dir, output_prosody_dir)
