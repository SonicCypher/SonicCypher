# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# import numpy as np
# import tensorflow as tf
# from ref_encoder import reference_encoder  
# from hyperparams import Hyperparams as hp

# # Paths for mel and prosody files
# mel_dir = "./mels"
# prosody_dir = "./prosody"

# # Ensure output directory exists
# os.makedirs(prosody_dir, exist_ok=True)

# # Placeholder for input mel spectrograms
# mel_input = tf.placeholder(dtype=tf.float32, shape=[None, None, None])  # Adjust shape as needed
# mel_input = tf.expand_dims(mel_input, axis=-1) 
# # Generate prosody embeddings using the reference encoder
# prosody_embeddings = reference_encoder(mel_input, is_training=False)

# # Start a TensorFlow session
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())  # Initialize variables

#     for mel_file in os.listdir(mel_dir):
#         if mel_file.endswith("_mel.npy"):
#             mel_path = os.path.join(mel_dir, mel_file)
#             mel_data = np.load(mel_path)  # Load mel spectrogram
            
#             mel_data = np.expand_dims(mel_data, axis=0)  # Add batch dimension
            
#             # Run the TensorFlow graph to compute embeddings
#             prosody_embedding = sess.run(prosody_embeddings, feed_dict={mel_input: mel_data})

#             prosody_path = os.path.join(prosody_dir, mel_file.replace("_mel.npy", "_prosody.npy"))
#             np.save(prosody_path, prosody_embedding)  # Save output as numpy file

#             print(f"Saved prosody embedding: {prosody_path}")

# print("Prosody embedding generation complete.")
# import os
# import numpy as np
# import tensorflow as tf
# from get_spectograms import get_spectrograms  # Assuming the get_spectrograms function is in this module
# from ref_encoder import reference_encoder  # Assuming the reference_encoder function is in this module
# from hyperparams import Hyperparams as hp

# def process_and_generate_prosody(input_mel_dir, output_prosody_dir):
#     os.makedirs(output_prosody_dir, exist_ok=True)
    
#     # Loop through all files in the input directory
#     for file_name in os.listdir(input_mel_dir):
#         if file_name.endswith("_mel.npy"):
#             print(f"Processing {file_name}...")
#             mel_path = os.path.join(input_mel_dir, file_name)
            
#             # Load the mel spectrogram
#             mel = np.load(mel_path)  # Shape (time_steps, n_mels)

#             # Reshape the mel spectrogram to 4D tensor (N, T, W, C)
#             N = 1  # Batch size of 1 (one file at a time)
#             T, W = mel.shape  # T = time_steps, W = n_mels
#             C = 1  # Single channel (grayscale)
            
#             # Reshape to (1, T, W, 1)
#             mel_4d = mel.reshape(N, T, W, C)

#             # Convert to a TensorFlow tensor
#             mel_tensor = tf.convert_to_tensor(mel_4d, dtype=tf.float32)

#             # Generate prosody embedding using reference_encoder
#             prosody_embedding = reference_encoder(mel_tensor, is_training=False)  # Set is_training to False during inference

#             # Save the prosody embedding
#             prosody_path = os.path.join(output_prosody_dir, file_name.replace("_mel.npy", "_prosody.npy"))
#             np.save(prosody_path, prosody_embedding.numpy())  # Convert TensorFlow tensor to numpy array and save

#             print(f"Finished processing {file_name}, prosody saved at {prosody_path}.")

# # Define the directories
# input_mel_dir = "./mels"  # Directory containing mel spectrograms
# output_prosody_dir = "./prosody_embeddings"  # Directory to save the prosody embeddings

# # Run the process and generate prosody embeddings
# process_and_generate_prosody(input_mel_dir, output_prosody_dir)
#///////////////////////////////////////////////

# import os
# import numpy as np
# import tensorflow as tf
# from ref_encoder import reference_encoder  # Assuming the reference_encoder function is in this module
# from hyperparams import Hyperparams as hp

# CHECKPOINT_PATH = "./logdir/ref_encoder.ckpt-5000"

# def process_and_generate_prosody(input_mel_dir, output_prosody_dir):
#     os.makedirs(output_prosody_dir, exist_ok=True)
#     print("entered process_and_generate_prosody")

#     processed_files = []
#     failed_files = []

#     for file_name in os.listdir(input_mel_dir):
#         print(f"file_name: {file_name}")
#         if file_name.endswith("_mel.npy"):
#             print(f"Processing {file_name}...da da da")

#             try:
#                 print("entered try block")
#                 mel_path = os.path.join(input_mel_dir, file_name)
#                 mel = np.load(mel_path)  # Shape (time_steps, n_mels)

#                 # Reshape mel spectrogram to 4D tensor (N, T, W, C)
#                 N, (T, W), C = 1, mel.shape, 1
#                 mel_4d = mel.reshape(N, T, W, C)
#                 print(f"mel_4d shape: {mel_4d.shape}")

#                 mel_tensor = tf.convert_to_tensor(mel_4d, dtype=tf.float32)
                
#                 audio_name = file_name.replace("_mel.npy", "")
#                 print(f"audio_name: {audio_name}")

#                 # Generate prosody embedding
#                 prosody_embedding = reference_encoder(mel_tensor, is_training=False, audio_names=[audio_name], save_dir=output_prosody_dir)

#                 with tf.Session() as sess:
#                     prosody_embedding_np = sess.run(prosody_embedding)
#                 # Generate prosody embedding
#                 #prosody_embedding = reference_encoder(mel_tensor, is_training=False)

#                 # Save prosody embedding
#                 prosody_path = os.path.join(output_prosody_dir, file_name.replace("_mel.npy", "_prosody.npy"))
#                 np.save(prosody_path, prosody_embedding_np)

#                 processed_files.append(file_name)
#                 print(f"Finished processing {file_name}, prosody saved at {prosody_path}.")
#             except Exception as e:
#                 failed_files.append(file_name)
#                 print(f"Error processing {file_name}: {e}")

#     # Verification
#     print("\n--- Processing Summary ---")
#     print(f"Total files processed successfully: {len(processed_files)}")
#     print(f"Total files failed: {len(failed_files)}")

#     if failed_files:
#         print("Failed files:", failed_files)

# # Define directories
# input_mel_dir = "./mels"
# output_prosody_dir = "./prosody_embeddings"

# # Run the function
# process_and_generate_prosody(input_mel_dir, output_prosody_dir)


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf  # Pure TensorFlow 1.x

from ref_encoder import reference_encoder  # Import reference encoder function
from hyperparams import Hyperparams as hp

# Path to the saved model checkpoint
CHECKPOINT_PATH = "./logdir/ref_encoder_updated.ckpt"

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
        # Saver to restore checkpoint
        saver = tf.train.Saver()
        # Restore the trained weights
        print(f"Loading model from {CHECKPOINT_PATH}...")
        saver.restore(sess, CHECKPOINT_PATH)
        print("Model loaded successfully!")

        for file_name in os.listdir(input_mel_dir):
            print(f"file_name: {file_name}")
            if file_name.endswith("_mel.npy"):
                print(f"Processing {file_name}...")

                try:
                    mel_path = os.path.join(input_mel_dir, file_name)
                    mel = np.load(mel_path)  # Load mel spectrogram

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
input_mel_dir = "./mels"
output_prosody_dir = "./prosody_embeddings"

# Run the function
process_and_generate_prosody(input_mel_dir, output_prosody_dir)

# import os
# import numpy as np
# import tensorflow as tf  # Pure TensorFlow 1.x

# from ref_encoder import reference_encoder  # Import reference encoder function
# from hyperparams import Hyperparams as hp

# # Path to the saved model checkpoint
# CHECKPOINT_PATH = "./logdir/ref_encoder_updated.ckpt"

# def process_and_generate_prosody(input_mel_dir, output_prosody_dir):
#     os.makedirs(output_prosody_dir, exist_ok=True)
#     print("entered process_and_generate_prosody")

#     processed_files = []
#     failed_files = []
    
#     # Reset graph before restoring weights
#     tf.reset_default_graph()

#     # Define input placeholder (assuming 80 mel bins)
#     mel_input = tf.placeholder(dtype=tf.float32, shape=[1, None, 80, 1], name="mel_input")

#     # Build the model
#     prosody_embedding = reference_encoder(mel_input, is_training=False)

#     # Create a custom mapping to align variable names
#     with tf.Session() as sess:
#         # List variables in the checkpoint
#         variables_in_checkpoint = tf.train.list_variables(CHECKPOINT_PATH)

#         # Manually create a mapping for checkpoint variables
#         variable_map = {}
#         for name, shape in variables_in_checkpoint:
#             # Adjust the variable name by removing 'net/' prefix from checkpoint
#             if name.startswith('net/reference_encoder/'):
#                 new_name = name.replace('net/reference_encoder/', 'reference_encoder/')
#                 print(f"Mapping: {name} -> {new_name}")
#                 variable_map[new_name] = name

#         # Restore the checkpoint with a direct assignment of the variables
#         saver = tf.train.Saver()

#         # Now load the checkpoint
#         print(f"Loading model from {CHECKPOINT_PATH}...")
#         saver.restore(sess, CHECKPOINT_PATH)
#         print("Model loaded successfully!")

#         # Assign variables manually from the checkpoint
#         for new_name, checkpoint_name in variable_map.items():
#             # Get the variable by its name in the model graph
#             variable = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, new_name)[0]
#             # Assign the value from the checkpoint variable to the model variable
#             checkpoint_value = tf.train.load_variable(CHECKPOINT_PATH, checkpoint_name)
#             assign_op = tf.assign(variable, checkpoint_value)
#             sess.run(assign_op)

#         for file_name in os.listdir(input_mel_dir):
#             print(f"file_name: {file_name}")
#             if file_name.endswith("_mel.npy"):
#                 print(f"Processing {file_name}...")

#                 try:
#                     mel_path = os.path.join(input_mel_dir, file_name)
#                     mel = np.load(mel_path)  # Load mel spectrogram

#                     # Reshape mel spectrogram to match input shape (N, T, 80, C)
#                     mel_4d = mel.reshape(1, mel.shape[0], mel.shape[1], 1)

#                     audio_name = file_name.replace("_mel.npy", "")
#                     print(f"Processing {audio_name}...")

#                     # Run session to get prosody embedding
#                     prosody_embedding_np = sess.run(prosody_embedding, feed_dict={mel_input: mel_4d})

#                     # Save prosody embedding
#                     prosody_path = os.path.join(output_prosody_dir, file_name.replace("_mel.npy", "_prosody.npy"))
#                     np.save(prosody_path, prosody_embedding_np)

#                     processed_files.append(file_name)
#                     print(f"Finished processing {file_name}, prosody saved at {prosody_path}.")
#                 except Exception as e:
#                     failed_files.append(file_name)
#                     print(f"Error processing {file_name}: {e}")

#     # Summary
#     print("\n--- Processing Summary ---")
#     print(f"Total files processed successfully: {len(processed_files)}")
#     print(f"Total files failed: {len(failed_files)}")
#     if failed_files:
#         print("Failed files:", failed_files)

# # Define directories
# input_mel_dir = "./mels"
# output_prosody_dir = "./prosody_embeddings"

# # Run the function
# process_and_generate_prosody(input_mel_dir, output_prosody_dir)
