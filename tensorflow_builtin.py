import tensorflow as tf
import tensorflow_addons as tfa

# Define input shape
input_shape = (100,)

# Define the Universal Transformer model
input_layer = tf.keras.layers.Input(shape=input_shape)
transformer_layer = tfa.seq2seq.UniversalTransformer(
    num_heads=8,
    key_dim=64,
    num_layers=4,
    attention_dropout=0.2,
    inner_dropout=0.2,
    output_dropout=0.2,
    norm_eps=1e-6,
    inner_activation='relu',
    output_activation=None,
    use_masking=True,
    return_sequences=True
)(input_layer)
output_layer = tf.keras.layers.Dense(
    1, activation='sigmoid')(transformer_layer)
model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
