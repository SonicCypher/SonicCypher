import tensorflow as tf
import numpy as np
from modules import bn, gru
import sys

# Load mel spectrogram from .npy file
def load_mel_spectrogram(npy_path):
    return np.load(npy_path)

def reference_encoder(inputs, is_training=True, scope="reference_encoder", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
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

        shape = tf.shape(tensor)
        N, W, C = shape[0], shape[2], shape[3]
        tensor = tf.reshape(tensor, (N, -1, W * C))

        tensor = gru(tensor, num_units=128, bidirection=False, scope="gru")
        tensor = tensor[:, -1, :]

        prosody = tf.layers.dense(tensor, 128, activation=tf.nn.tanh)
    
    return prosody

if __name__ == "__main__":
    mel_path = "mels/LA_T_1000137_mel.npy"
    mel_spectrogram = load_mel_spectrogram(mel_path)
    
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)  # Add batch dimension
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1) # Add channel dimension
    
    inputs = tf.placeholder(tf.float32, shape=[None, mel_spectrogram.shape[1], mel_spectrogram.shape[2], 1])
    prosody_embedding = reference_encoder(inputs, is_training=False)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        prosody_output = sess.run(prosody_embedding, feed_dict={inputs: mel_spectrogram})
        print("Prosody Embedding:", prosody_output)
