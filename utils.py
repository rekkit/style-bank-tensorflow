import tensorflow as tf


def gram_matrix(x):
    z = tf.reshape(x, [-1, tf.shape(x)[-1]])  # this makes z [H*W, C]
    z = tf.matmul(tf.transpose(z), z) / tf.cast(tf.shape(z)[0], dtype=tf.float32)

    return z

