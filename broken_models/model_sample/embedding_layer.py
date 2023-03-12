import tensorflow as tf


class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, input_vocab_size, embedding_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = tf.keras.layers.Embedding(
            input_vocab_size, embedding_dim)

    def call(self, inputs):
        return self.embedding(inputs)
