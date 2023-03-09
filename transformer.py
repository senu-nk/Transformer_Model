import numpy as np
import tensorflow as tf
from .tokenizer import Tokenizer
from .encoder_layer import EncoderLayer
from .decoder_layer import DecoderLayer
from .positional_encoding_layer import PositionalEncodingLayer


class Transformer(tf.keras.Model):

    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size,

                 pe_input, pe_target, rate=0.1):

        super(Transformer, self).__init__()

        self.tokenizer = Tokenizer(input_vocab_size)

        self.num_layers = num_layers

        self.d_model = d_model

        self.num_heads = num_heads

        self.dff = dff

        self.input_vocab_size = input_vocab_size

        self.target_vocab_size = target_vocab_size

        self.rate = rate

        self.embedding_input = tf.keras.layers.Embedding(
            input_vocab_size, d_model)

        self.embedding_target = tf.keras.layers.Embedding(
            target_vocab_size, d_model)

        self.pos_encoding_input = PositionalEncodingLayer(pe_input, d_model)

        self.pos_encoding_target = PositionalEncodingLayer(pe_target, d_model)

        self.dropout = tf.keras.layers.Dropout(rate)

        self.encoder = [EncoderLayer(d_model, num_heads, dff, rate)

                        for _ in range(num_layers)]

        self.decoder = [DecoderLayer(d_model, num_heads, dff, rate)

                        for _ in range(num_layers)]

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs):

        input_seq, target_seq, training, enc_padding_mask, look_ahead_mask, dec_padding_mask = inputs

        input_seq = tf.cast(input_seq, dtype=tf.int32)

        target_seq = tf.cast(target_seq, dtype=tf.int32)

        # embedding and positional encoding for input sequence

        input_seq = self.embedding_input(input_seq)

        input_seq *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        input_seq = self.pos_encoding_input(input_seq)

        # apply dropout to input sequence

        input_seq = self.dropout(input_seq, training=training)

        # encoding input sequence

        for i in range(self.num_layers):

            input_seq = self.encoder[i]([input_seq, enc_padding_mask])

        # embedding and positional encoding for target sequence

        target_seq = self.embedding_target(target_seq)

        target_seq *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        target_seq = self.pos_encoding_target(target_seq)

        # apply dropout to target sequence

        target_seq = self.dropout(target_seq, training=training)

        # decoding target sequence

        for i in range(self.num_layers):

            target_seq, attention_weights = self.decoder[i](
                [target_seq, input_seq, look_ahead_mask, dec_padding_mask])

        # final dense layer

        output = self.final_layer(target_seq)

        return output
