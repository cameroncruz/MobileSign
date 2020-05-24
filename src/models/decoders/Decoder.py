import tensorflow as tf


class Decoder(tf.keras.Model):
    def __init__(self, hidden_size, vocab_size, embedding_dim):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.query = tf.keras.layers.Dense(units=hidden_size)
        self.value = tf.keras.layers.Dense(units=hidden_size)
        self.attention = tf.keras.layers.Attention(causal=True)
        self.lstm = tf.keras.layers.LSTM(
            units=hidden_size, return_sequences=False, return_state=True
        )
        self.classifier = tf.keras.layers.Dense(units=vocab_size, activation="softmax")

    def call(self, inputs, training=None, mask=None):
        prev_word, features, memory_state, carry_state = inputs
        x = self.embedding(prev_word)
        context = self.attention(
            [self.query(features), self.value(tf.expand_dims(memory_state, 1))]
        )
        x = tf.concat([tf.expand_dims(tf.reduce_sum(context, axis=1), 1), x], axis=-1)
        output, memory_state, carry_state = self.lstm(
            x, initial_state=[memory_state, carry_state]
        )
        preds = self.classifier(output)
        return preds, memory_state, carry_state

    def reset_state(self, batch_size):
        return (
            tf.zeros((batch_size, self.hidden_size)),
            tf.zeros((batch_size, self.hidden_size)),
        )
