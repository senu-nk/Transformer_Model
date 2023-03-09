import tensorflow as tf
from .layer_normalization import LayerNormalization


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model)
        self.mha2 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([tf.keras.layers.Dense(dff, activation='relu'),
                                        tf.keras.layers.Dense(d_model)])
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.layernorm3 = LayerNormalization()
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, enc_output, training=False):
        # masked multi-head attention layer with look-ahead mask
        attn1 = self.mha1(inputs, inputs, inputs,
                          mask=self.create_look_ahead_mask(tf.shape(inputs)[1]))
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(inputs + attn1)

        # multi-head attention layer with padding mask
        attn2 = self.mha2(enc_output, enc_output, out1,
                          mask=self.create_padding_mask(enc_output))
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        # feedforward layer
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3

    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

    def create_padding_mask(self, seq):
        mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]
