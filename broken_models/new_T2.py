import tensorflow as tf
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy


class UniversalTransformer:

    def __init__(self, num_layers, hidden_size, num_heads, max_seq_len, vocab_size, dropout_rate=0.1):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate

    def build(self):
        # Input layer
        inputs = Input(shape=(self.max_seq_len,), dtype=tf.int32)

        # Embedding layer
        embeddings = Embedding(input_dim=self.vocab_size,
                               output_dim=self.hidden_size)(inputs)

        # Encoder
        x = embeddings
        for i in range(self.num_layers):
            x = self.add_encoder_layer(x, i)

        # Decoder
        decoder_inputs = Input(shape=(self.max_seq_len,), dtype=tf.int32)
        decoder_embeddings = Embedding(
            input_dim=self.vocab_size, output_dim=self.hidden_size)(decoder_inputs)
        x = decoder_embeddings
        for i in range(self.num_layers):
            x = self.add_decoder_layer(x, embeddings)
        outputs = Dense(units=self.vocab_size, activation='softmax')(x)

        # Build model
        model = Model(inputs=[inputs, decoder_inputs], outputs=outputs)

        # Compile model
        optimizer = Adam(learning_rate=0.001)
        loss_fn = SparseCategoricalCrossentropy()
        model.compile(optimizer=optimizer, loss=loss_fn)

        return model

    def add_encoder_layer(self, x, layer_num):
        # Multi-Head Attention
        attention_output = self.add_multihead_attention_layer(x, x, x)
        # Add & Norm
        x = self.add_and_norm(x, attention_output)
        # Feed-Forward
        ff_output = self.add_feedforward_layer(x)
        # Add & Norm
        x = self.add_and_norm(x, ff_output)
        return x

    def add_decoder_layer(self, x, encoder_output):
        # Masked Multi-Head Attention
        masked_attention_output = self.add_masked_multihead_attention_layer(
            x, x, x)
        # Add & Norm
        x = self.add_and_norm(x, masked_attention_output)
        # Multi-Head Attention
        attention_output = self.add_multihead_attention_layer(
            x, encoder_output, encoder_output)
        # Add & Norm
        x = self.add_and_norm(x, attention_output)
        # Feed-Forward
        ff_output = self.add_feedforward_layer(x)
        # Add & Norm
        x = self.add_and_norm(x, ff_output)
        return x

    def add_multihead_attention_layer(self, queries, keys, values):
        # Calculate attention scores
        queries = Dense(units=self.hidden_size)(queries)
        keys = Dense(units=self.hidden_size)(keys)
        values = Dense(units=self.hidden_size)(values)
        scores = tf.matmul(queries, keys, transpose_b=True)
        # Scale attention scores
        scores = scores / (self.hidden_size ** 0.5)
        # Mask padding tokens
        mask = tf.math.equal(tf.reduce_sum(keys, axis=-1), 0)
        mask = tf.expand_dims(mask, axis=1)
        mask = tf.tile(mask, multiples=[1, self.max_seq_len, 1])
        # Set padding tokens to -inf
        padding_mask = tf.ones_like(scores) * -1e9
        scores = tf.where(mask, padding_mask, scores)
        # Apply softmax to attention scores
        attention_probs = tf.nn.softmax(scores, axis=-1)
        # Apply dropout to attention probabilities
        attention_probs = tf.keras.layers.Dropout(
            self.dropout_rate)(attention_probs)
        # Calculate attention output
        attention_output = tf.matmul(attention_probs, values)
        return attention_output

    def add_masked_multihead_attention_layer(self, queries, keys, values):
        # Calculate attention scores
        queries = Dense(units=self.hidden_size)(queries)
        keys = Dense(units=self.hidden_size)(keys)
        values = Dense(units=self.hidden_size)(values)
        scores = tf.matmul(queries, keys, transpose_b=True)
        # Scale attention scores
        scores = scores / (self.hidden_size ** 0.5)
        # Apply masking to attention scores
        mask = tf.linalg.LinearOperatorLowerTriangular(
            tf.ones_like(scores)).to_mask()
        mask = tf.expand_dims(mask, axis=1)
        mask = tf.tile(mask, multiples=[1, self.max_seq_len, 1])
        padding_mask = tf.ones_like(scores) * -1e9
        scores = tf.where(mask, scores, padding_mask)
        # Apply softmax to attention scores
        attention_probs = tf.nn.softmax(scores, axis=-1)
        # Apply dropout to attention probabilities
        attention_probs = tf.keras.layers.Dropout(
            self.dropout_rate)(attention_probs)
        # Calculate attention output
        attention_output = tf.matmul(attention_probs, values)
        return attention_output

    def add_feedforward_layer(self, x):
        x = Dense(units=self.hidden_size, activation='relu')(x)
        x = Dense(units=self.hidden_size)(x)
        return x

    def add_and_norm(self, x1, x2):
        x = tf.keras.layers.Add()([x1, x2])
        x = tf.keras.layers.LayerNormalization()(x)
        return x
