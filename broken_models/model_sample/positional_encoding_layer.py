import numpy as np
import tensorflow as tf


class PositionalEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, max_seq_len, embedding_dim):
        super(PositionalEncodingLayer, self).__init__()
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.pos_encoding = self.get_positional_encoding()

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def get_positional_encoding(self):
        angle_rads = self.get_angles(np.arange(self.max_seq_len)[:, np.newaxis],
                                     np.arange(self.embedding_dim)[
            np.newaxis, :],
            self.embedding_dim)

        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])

        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
