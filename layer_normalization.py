import tensorflow as tf


class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6):
        super(LayerNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name="gamma", shape=input_shape[-1:], initializer=tf.ones_initializer(), trainable=True)
        self.beta = self.add_weight(
            name="beta", shape=input_shape[-1:], initializer=tf.zeros_initializer(), trainable=True)

    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        variance = tf.reduce_mean(
            tf.square(inputs - mean), axis=-1, keepdims=True)
        norm_inputs = (inputs - mean) * tf.math.rsqrt(variance + self.epsilon)
        outputs = self.gamma * norm_inputs + self.beta
        return outputs
