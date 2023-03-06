import tensorflow as tf


class Tokenizer:
    def __init__(self, num_words):
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=num_words, oov_token="<OOV>")

    def fit_on_texts(self, texts):
        self.tokenizer.fit_on_texts(texts)

    def texts_to_sequences(self, texts):
        return self.tokenizer.texts_to_sequences(texts)

    def sequences_to_texts(self, sequences):
        return self.tokenizer.sequences_to_texts(sequences)

    def get_vocab_size(self):
        return len(self.tokenizer.word_index) + 1
