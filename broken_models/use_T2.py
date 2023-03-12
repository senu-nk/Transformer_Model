import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from broken_models.new_T2 import UniversalTransformer

# Set hyperparameters
num_samples = 1000
input_dim = 10
output_dim = 5
max_seq_len = 20
num_layers = 2
hidden_size = 64
num_heads = 8
dropout_rate = 0.1
batch_size = 32
epochs = 10

# Generate random dataset
inputs = np.random.rand(num_samples, max_seq_len, input_dim)
outputs = np.random.rand(num_samples, max_seq_len, output_dim)

# Build model
input_layer = Input(shape=(max_seq_len, input_dim))
output_layer = Input(shape=(max_seq_len, output_dim))
universal_transformer = UniversalTransformer(
    num_layers, hidden_size, num_heads, max_seq_len, dropout_rate)
encoder_output = universal_transformer.add_encoder_layer(input_layer)
decoder_output = universal_transformer.add_decoder_layer(
    output_layer, encoder_output)
model = Model(inputs=[input_layer, output_layer], outputs=decoder_output)
model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(x=[inputs, outputs], y=outputs, batch_size=batch_size, epochs=epochs)
