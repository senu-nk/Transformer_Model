# Imports
import nltk
import string
import pandas as pd
import tensorflow as tf
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from Transformer.transformer import Transformer
# %%
# Reading data

df = pd.read_csv("IMDB Dataset.csv")
df.head()

# %%
# Pre-processing fucntions


def clear_html(text):
    return BeautifulSoup(text, "html.parser").get_text()


def clear_punctuations(text):
    return text.translate(str.maketrans("", "", string.punctuation))


def sentiment_converter(text):
    if text == "positive":
        return 1
    else:
        return 0
# %% Pre-processing part 1


df["review"] = df["review"].apply(clear_html)
df["review"] = df["review"].apply(clear_punctuations)
df["sentiment"] = df["sentiment"].apply(sentiment_converter)
df.head()

# %% Pre-processing part2 tokenzing and spliting

tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=10_000, oov_token='<OOV>')
tokenizer.fit_on_texts(df["review"])
sequences = tokenizer.texts_to_sequences(df["review"])
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
    sequences, maxlen=128, padding="post", truncating="post")

X_train, X_val, y_train, y_val = train_test_split(
    padded_sequences, df["sentiment"], test_size=0.2, random_state=42)
