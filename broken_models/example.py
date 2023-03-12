import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf
from model_sample.transformer import Transformer

# Define some hyperparameters
num_layers = 4
d_model = 128
num_heads = 8
dff = 512
dropout_rate = 0.1

# Instantiate the Transformer model
transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size=8192, target_vocab_size=8192,
                          pe_input=1000, pe_target=1000, rate=dropout_rate)

# Load the Multi30k dataset
data, info = tfds.load('multi30k', with_info=True, as_supervised=True)

# Extract the tokenizer for the English and French languages
en_tokenizer = info.features['en'].encoder
fr_tokenizer = info.features['de'].encoder

# Define a function to preprocess the dataset


def preprocess_sentence(sentence):
    # Tokenize the sentence
    sentence = en_tokenizer.encode(sentence.numpy().decode('utf-8'))
    # Add a start and end token
    sentence = [en_tokenizer.vocab_size] + \
        sentence + [en_tokenizer.vocab_size + 1]
    return sentence

# Define a function to encode a sentence as a tensor


def encode_sentence(sentence):
    # Preprocess the sentence
    sentence = preprocess_sentence(sentence)
    # Convert the sentence to a tensor
    sentence = tf.keras.preprocessing.sequence.pad_sequences(
        [sentence], maxlen=40, padding='post')
    return sentence


# Define some input tensors
input_sentence = "the cat sat on the mat"
input_tensor = encode_sentence(input_sentence)
target_tensor = tf.constant([[fr_tokenizer.vocab_size]])

# Define the masks
input_padding_mask = tf.zeros((1, 1, 1, 40))
lookahead_mask = transformer.create_look_ahead_mask(1, 40)
target_padding_mask = tf.zeros((1, 1, 1, 1))

# Generate the output sentence
for i in range(40):
    # Call the model on the inputs
    predictions = transformer([input_tensor, target_tensor, True,
                               input_padding_mask, lookahead_mask, target_padding_mask])
    # Get the last predicted word
    prediction = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(prediction, axis=-1), tf.int32)
    # Append the predicted word to the target tensor
    target_tensor = tf.concat([target_tensor, predicted_id], axis=-1)
    # Stop if the end token is predicted
    if predicted_id == fr_tokenizer.vocab_size + 1:
        break

# Decode the output sentence
output_sentence = fr_tokenizer.decode(target_tensor.numpy().squeeze())
print(output_sentence)
