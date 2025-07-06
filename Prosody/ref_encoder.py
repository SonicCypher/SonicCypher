import tensorflow as tf
import numpy as np
import os
from modules import bn, gru

def reference_encoder(inputs, is_training=True, scope="reference_encoder", reuse=None, audio_names=None, save_dir='./prosody_embeddings'):
    print("entered reference_encoder---1")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        print("entered reference_encoder---2")
        # 6-Layer Strided Conv2D -> (N, T/64, n_mels/64, 128)
        tensor = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=3, strides=2, padding='SAME')
        tensor = bn(tensor, is_training=is_training, activation_fn=tf.nn.relu, scope="bn1")

        tensor = tf.layers.conv2d(inputs=tensor, filters=32, kernel_size=3, strides=2, padding='SAME')
        tensor = bn(tensor, is_training=is_training, activation_fn=tf.nn.relu, scope="bn2")

        tensor = tf.layers.conv2d(inputs=tensor, filters=64, kernel_size=3, strides=2, padding='SAME')
        tensor = bn(tensor, is_training=is_training, activation_fn=tf.nn.relu, scope="bn3")

        tensor = tf.layers.conv2d(inputs=tensor, filters=64, kernel_size=3, strides=2, padding='SAME')
        tensor = bn(tensor, is_training=is_training, activation_fn=tf.nn.relu, scope="bn4")

        tensor = tf.layers.conv2d(inputs=tensor, filters=128, kernel_size=3, strides=2, padding='SAME')
        tensor = bn(tensor, is_training=is_training, activation_fn=tf.nn.relu, scope="bn5")

        tensor = tf.layers.conv2d(inputs=tensor, filters=128, kernel_size=3, strides=2, padding='SAME')
        tensor = bn(tensor, is_training=is_training, activation_fn=tf.nn.relu, scope="bn6")

        batch_size = tf.shape(tensor)[0]
        _, _, W, C = tensor.get_shape().as_list()
        tensor = tf.reshape(tensor, [batch_size, -1, W*C])

        # GRU -> (N, T/64, 128) -> (N, 128)
        tensor = gru(tensor, num_units=128, bidirection=False, scope="gru")
        tensor = tensor[:, -1, :]

        # FC -> (N, 128)
        prosody = tf.layers.dense(tensor, 128, activation=tf.nn.tanh)
        
        if audio_names is not None:
            with tf.compat.v1.Session() as sess:
                print("entered session")
                sess.run(tf.compat.v1.global_variables_initializer())

                for i, audio_name in enumerate(audio_names):
                    print(f"Processing {audio_name}...")
                    embedding = sess.run(prosody[i])  # Convert tensor to NumPy array

                    save_path = os.path.join(save_dir, f"{audio_name}.bin")
                    tf.io.write_file(save_path, tf.io.encode_base64(tf.convert_to_tensor(embedding.tobytes())))
                    
        print("exited session")
                
    return prosody
    #     # Unroll -> (N, T/64, 128*n_mels/64)
    #     N, _, W, C = tensor.get_shape().as_list()
    #     tensor = tf.reshape(tensor, (N, -1, W*C))

    #     # GRU -> (N, T/64, 128) -> (N, 128)
    #     tensor = gru(tensor, num_units=128, bidirection=False, scope="gru")
    #     tensor = tensor[:, -1, :]

    #     # FC -> (N, 128)
    #     prosody = tf.layers.dense(tensor, 128, activation=tf.nn.tanh)
        
    #     if audio_names is not None:
    #         with tf.Session() as sess:
    #             print("entered session")
    #             sess.run(tf.global_variables_initializer())

    #             for i, audio_name in enumerate(audio_names):
    #                 print(f"Processing {audio_name}...")
    #                 embedding = sess.run(prosody[i])  # Convert tensor to NumPy array

    #                 save_path = os.path.join(save_dir, f"{audio_name}.bin")
    #                 tf.io.write_file(save_path, tf.io.encode_base64(tf.convert_to_tensor(embedding.tobytes())))
                    
    #     print("exited session")

                
    # return prosody
