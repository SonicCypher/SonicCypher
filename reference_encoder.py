
# reference_encoder.py
def reference_encoder(inputs, is_training=True, scope="encoder", reuse=None):
    '''
    Args:
      inputs: A 3d tensor with shape of (N, Ty, n_mels), with dtype of float32.
                Melspectrogram of reference audio.
      is_training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      Prosody vectors. Has the shape of (N, 128).
    '''
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
