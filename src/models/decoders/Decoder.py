import tensorflow as tf


class Decoder(tf.keras.Model):
    def __init__(self, hidden_size, vocab_size):
        super(Decoder, self).__init__()
        self.query = tf.keras.layers.Dense(units=512, activation=None)
        self.value = tf.keras.layers.Dense(units=512, activation=None)
        self.attention = tf.keras.layers.Attention(use_scale=True)
        self.lstm = tf.keras.layers.LSTM(units=hidden_size, return_sequences=True)
        self.classifier = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=vocab_size, activation='softmax'))

    def call(self, inputs, training=None, mask=None):
        x = self.attention([self.query(inputs), self.value(inputs)])
        x = self.lstm(x)
        preds = self.classifier(x)
        return preds
