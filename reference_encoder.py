import tensorflow as tf
from modules import bn, gru
# reference_encoder.py
def reference_encoder(inputs, is_training=True, scope="encoder", reuse=None):
    
    with tf.variable_scope(scope, reuse=reuse):
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

        # Unroll -> (N, T/64, 128*n_mels/64)
        N, _, W, C = tensor.get_shape().as_list()
        tensor = tf.reshape(tensor, (N, -1, W*C))

        # GRU -> (N, T/64, 128) -> (N, 128)
        tensor = gru(tensor, num_units=128, bidirection=False, scope="gru")
        tensor = tensor[:, -1, :]

        # FC -> (N, 128)
        prosody = tf.layers.dense(tensor, 128, activation=tf.nn.tanh)

    return prosody
