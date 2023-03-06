import tensorflow as tf


class FeedForwardLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, dropout_rate):
        super(FeedForwardLayer, self).__init__()
        self.dense1 = tf.keras.layers.Dense(d_ff, activation='relu')
        self.dense2 = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)
        return x
